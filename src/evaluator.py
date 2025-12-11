"""
Offline Evaluation Framework for Contextual Bandits.

This module implements offline evaluation using the Replay Method, which simulates
online learning by processing historical rating data in temporal order.

Classes:
    OfflineEvaluator: Main evaluation framework
    RandomBaseline: Random recommendation baseline
    PopularityBaseline: Popularity-based recommendation baseline

Example:
    >>> from src.evaluator import OfflineEvaluator
    >>> from src.linucb import LinUCB
    >>>
    >>> bandit = LinUCB(n_arms=3883, n_features=48, alpha=1.0)
    >>> evaluator = OfflineEvaluator(
    ...     bandit, user_features, movie_features,
    ...     movie_id_to_idx, movie_idx_to_id
    ... )
    >>> results = evaluator.replay_evaluation(ratings, max_rounds=10000)
    >>> evaluator.plot_learning_curve()
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
from collections import defaultdict


class OfflineEvaluator:
    """
    Offline evaluation framework using Replay Method.

    Simulates online learning by processing historical rating data in temporal
    order and only updating when the algorithm's recommendation matches the
    user's actual choice.

    Attributes:
        bandit: The bandit algorithm to evaluate
        user_features: Dictionary mapping UserID to feature vectors
        movie_features: Dictionary mapping MovieID to feature vectors
        movie_id_to_idx: Dictionary mapping MovieID to array indices (for bandit)
        movie_idx_to_id: Dictionary mapping array indices to MovieID (for result interpretation)
        reward_threshold: Rating threshold for positive reward
        total_rounds: Total number of evaluation rounds
        matched_rounds: Number of rounds where recommendation matched actual choice
        cumulative_reward: Sum of all received rewards
        reward_history: List of cumulative rewards over time
        ctr_history: List of CTR values over time
    """

    def __init__(
        self,
        bandit_algorithm,
        user_features: Dict[int, np.ndarray],
        movie_features: Dict[int, np.ndarray],
        movie_id_to_idx: Dict[int, int],
        movie_idx_to_id: Dict[int, int],
        reward_threshold: float = 4.0
    ):
        """
        Initialize evaluator.

        Args:
            bandit_algorithm: Instance of a bandit algorithm (e.g., LinUCB)
            user_features: Dictionary mapping UserID to feature vector
            movie_features: Dictionary mapping MovieID to feature vector
            movie_id_to_idx: Mapping from MovieID to array index for bandit
            movie_idx_to_id: Mapping from array index to MovieID
            reward_threshold: Threshold for converting ratings to binary rewards

        Raises:
            ValueError: If reward_threshold is not in [1, 5]
        """
        if not (1 <= reward_threshold <= 5):
            raise ValueError(f"reward_threshold must be in [1, 5], got {reward_threshold}")

        self.bandit = bandit_algorithm
        self.user_features = user_features
        self.movie_features = movie_features
        self.movie_id_to_idx = movie_id_to_idx
        self.movie_idx_to_id = movie_idx_to_id
        self.reward_threshold = reward_threshold

        # Metrics tracking
        self.total_rounds = 0
        self.matched_rounds = 0
        self.cumulative_reward = 0.0
        self.reward_history = []
        self.ctr_history = []
        self.time_points = []

    def replay_evaluation(
        self,
        ratings_df: pd.DataFrame,
        max_rounds: Optional[int] = None,
        candidate_strategy: str = 'random_k',
        k_candidates: int = 20,
        verbose: bool = True,
        record_interval: int = 1000
    ) -> Dict[str, Any]:
        """
        Run offline evaluation using replay method.

        Args:
            ratings_df: DataFrame with ratings sorted by timestamp
            max_rounds: Maximum number of rounds to simulate (None = all data)
            candidate_strategy: Strategy for generating candidates ('random_k', 'all')
            k_candidates: Number of candidates for 'random_k' strategy
            verbose: Whether to print progress
            record_interval: Interval for recording metrics

        Returns:
            Dictionary with evaluation metrics

        Raises:
            ValueError: If ratings_df is empty or missing required columns
        """
        required_columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
        if not all(col in ratings_df.columns for col in required_columns):
            raise ValueError(f"ratings_df must contain columns: {required_columns}")

        if len(ratings_df) == 0:
            raise ValueError("ratings_df cannot be empty")

        n_rounds = len(ratings_df) if max_rounds is None else min(max_rounds, len(ratings_df))

        for i, (_, row) in enumerate(ratings_df.iloc[:n_rounds].iterrows()):
            if i >= n_rounds:
                break

            user_id = int(row['UserID'])
            actual_movie_id = int(row['MovieID'])
            actual_rating = float(row['Rating'])

            # Skip if user or movie features not available
            if user_id not in self.user_features:
                continue
            if actual_movie_id not in self.movie_features:
                continue
            if actual_movie_id not in self.movie_id_to_idx:
                continue

            # Get features
            user_vec = self.user_features[user_id]
            actual_movie_vec = self.movie_features[actual_movie_id]

            # Generate candidate set
            if candidate_strategy == 'all':
                candidate_movie_ids = list(self.movie_features.keys())
            elif candidate_strategy == 'random_k':
                candidate_movie_ids = self._generate_random_candidates(
                    actual_movie_id,
                    k_candidates
                )
            else:
                raise ValueError(f"Unknown candidate_strategy: {candidate_strategy}")

            # Convert MovieIDs to array indices
            candidate_indices = [
                self.movie_id_to_idx[mid]
                for mid in candidate_movie_ids
                if mid in self.movie_id_to_idx
            ]

            if len(candidate_indices) == 0:
                continue

            # Detect bandit type and select arm accordingly
            # Hybrid LinUCB uses separate shared and arm features
            # Disjoint LinUCB uses concatenated features
            is_hybrid = hasattr(self.bandit, 'shared_dim')

            if is_hybrid:
                # Hybrid mode: pass shared context (user) and arm contexts (movies)
                shared_context = user_vec
                arm_contexts = {
                    idx: self.movie_features[self.movie_idx_to_id[idx]]
                    for idx in candidate_indices
                }
                selected_idx, _ = self.bandit.select_arm(
                    shared_context,
                    arm_contexts,
                    candidate_arms=candidate_indices
                )
            else:
                # Disjoint mode
                if isinstance(self.bandit, RandomBaseline):
                    # Allow true random choice across the candidate set
                    context = self._create_context(user_vec, actual_movie_vec)
                    selected_idx, _ = self.bandit.select_arm(
                        context,
                        candidate_arms=candidate_indices
                    )
                else:
                    # Build context per candidate movie for LinUCB
                    best_idx = None
                    best_ucb = -float("inf")

                    for idx in candidate_indices:
                        movie_id = self.movie_idx_to_id[idx]
                        movie_vec = self.movie_features.get(movie_id)
                        if movie_vec is None:
                            continue

                        context = self._create_context(user_vec, movie_vec)
                        _, ucb = self.bandit.select_arm(
                            context,
                            candidate_arms=[idx]
                        )

                        if ucb > best_ucb:
                            best_ucb = ucb
                            best_idx = idx

                    if best_idx is None:
                        continue

                    selected_idx = best_idx

            # Convert back to MovieID
            recommended_movie_id = self.movie_idx_to_id[selected_idx]

            self.total_rounds += 1

            # Check if recommendation matches actual choice (Replay Method)
            if recommended_movie_id == actual_movie_id:
                self.matched_rounds += 1
                reward = self._rating_to_reward(actual_rating)
                self.cumulative_reward += reward

                # Update bandit with observed reward
                if is_hybrid:
                    # Hybrid mode: update with separate shared and arm contexts
                    shared_context = user_vec
                    arm_context = actual_movie_vec
                    self.bandit.update(selected_idx, shared_context, arm_context, reward)
                else:
                    # Disjoint mode: update with concatenated context
                    context = self._create_context(user_vec, actual_movie_vec)
                    self.bandit.update(selected_idx, context, reward)

            # Record metrics periodically
            if i % record_interval == 0 and i > 0:
                self._record_metrics(i)
                if verbose:
                    ctr = self.matched_rounds / self.total_rounds if self.total_rounds > 0 else 0
                    print(f"Round {i}/{n_rounds}: CTR={ctr:.4f}, "
                          f"Cumulative Reward={self.cumulative_reward:.2f}")

        # Final recording
        self._record_metrics(n_rounds)

        return self.compute_metrics()

    def _create_context(
        self,
        user_features: np.ndarray,
        movie_features: np.ndarray
    ) -> np.ndarray:
        """
        Combine user and movie features into context vector.

        Args:
            user_features: User feature vector
            movie_features: Movie feature vector

        Returns:
            Context vector (concatenation of user and movie features)
        """
        return np.concatenate([user_features, movie_features])

    def _generate_random_candidates(
        self,
        actual_movie: int,
        k: int
    ) -> List[int]:
        """
        Generate random candidate set including the actual movie.

        Args:
            actual_movie: The actual movie that was rated
            k: Number of candidates to generate

        Returns:
            List of MovieIDs including the actual movie
        """
        candidates = [actual_movie]

        # Get all available movies
        all_movies = list(self.movie_features.keys())

        # Remove actual movie from pool
        other_movies = [m for m in all_movies if m != actual_movie]

        # Sample random movies
        if len(other_movies) > 0:
            n_to_sample = min(k - 1, len(other_movies))
            random_movies = np.random.choice(
                other_movies,
                size=n_to_sample,
                replace=False
            )
            candidates.extend(random_movies.tolist())

        return candidates

    def _rating_to_reward(self, rating: float) -> float:
        """
        Convert rating to binary reward.

        Args:
            rating: User rating (1-5 stars)

        Returns:
            Binary reward: 1.0 if rating >= threshold, else 0.0
        """
        return 1.0 if rating >= self.reward_threshold else 0.0

    def _record_metrics(self, time_point: int) -> None:
        """
        Record current metrics to history.

        Args:
            time_point: Current time step
        """
        self.time_points.append(time_point)
        self.reward_history.append(self.cumulative_reward)
        ctr = self.matched_rounds / self.total_rounds if self.total_rounds > 0 else 0
        self.ctr_history.append(ctr)

    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute final evaluation metrics.

        Returns:
            Dictionary containing:
            - CTR: Click-through rate (match rate)
            - cumulative_reward: Total reward accumulated
            - average_reward: Average reward per match
            - total_rounds: Total evaluation rounds
            - matched_rounds: Number of matches
        """
        ctr = self.matched_rounds / self.total_rounds if self.total_rounds > 0 else 0
        avg_reward = (
            self.cumulative_reward / self.matched_rounds
            if self.matched_rounds > 0 else 0
        )

        return {
            'CTR': ctr,
            'cumulative_reward': self.cumulative_reward,
            'average_reward': avg_reward,
            'total_rounds': self.total_rounds,
            'matched_rounds': self.matched_rounds
        }

    def plot_learning_curve(
        self,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Plot cumulative reward learning curve.

        Args:
            save_path: Optional path to save the figure
            show: Whether to display the plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.time_points, self.reward_history, linewidth=2, color='#2E86AB')
        plt.xlabel('Time Steps', fontsize=12)
        plt.ylabel('Cumulative Reward', fontsize=12)
        plt.title('Learning Curve: Cumulative Reward Over Time',
                  fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    def plot_ctr_curve(
        self,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Plot CTR (Click-Through Rate) over time.

        Args:
            save_path: Optional path to save the figure
            show: Whether to display the plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.time_points, self.ctr_history, linewidth=2, color='#A23B72')
        plt.xlabel('Time Steps', fontsize=12)
        plt.ylabel('CTR (Click-Through Rate)', fontsize=12)
        plt.title('CTR Evolution Over Time', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()


class RandomBaseline:
    """
    Random recommendation baseline.

    Randomly selects an arm from the candidate set.
    """

    def __init__(self, n_arms: int, n_features: int):
        """
        Initialize random baseline.

        Args:
            n_arms: Number of arms (for API compatibility)
            n_features: Number of features (for API compatibility)
        """
        self.n_arms = n_arms
        self.n_features = n_features

    def select_arm(
        self,
        context: np.ndarray,
        candidate_arms: Optional[List[int]] = None
    ) -> Tuple[int, float]:
        """
        Randomly select an arm.

        Args:
            context: Context vector (ignored)
            candidate_arms: List of candidate arms

        Returns:
            Tuple of (selected_arm, score=0.0)
        """
        if candidate_arms is None:
            candidate_arms = list(range(self.n_arms))

        selected_arm = np.random.choice(candidate_arms)
        return int(selected_arm), 0.0

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        """
        Update (no-op for random baseline).

        Args:
            arm: Selected arm
            context: Context vector
            reward: Observed reward
        """
        pass  # Random baseline does not learn


class PopularityBaseline:
    """
    Popularity-based recommendation baseline.

    Recommends arms based on their historical selection frequency.
    """

    def __init__(self, n_arms: int, n_features: int):
        """
        Initialize popularity baseline.

        Args:
            n_arms: Number of arms
            n_features: Number of features (for API compatibility)
        """
        self.n_arms = n_arms
        self.n_features = n_features
        self.arm_counts = defaultdict(int)
        self.total_count = 0

    def select_arm(
        self,
        context: np.ndarray,
        candidate_arms: Optional[List[int]] = None
    ) -> Tuple[int, float]:
        """
        Select most popular arm from candidates.

        Args:
            context: Context vector (ignored)
            candidate_arms: List of candidate arms

        Returns:
            Tuple of (selected_arm, popularity_score)
        """
        if candidate_arms is None:
            candidate_arms = list(range(self.n_arms))

        # Find arm with highest count among candidates
        best_arm = max(
            candidate_arms,
            key=lambda arm: self.arm_counts[arm]
        )

        popularity_score = self.arm_counts[best_arm] / max(self.total_count, 1)

        return int(best_arm), float(popularity_score)

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        """
        Update arm popularity count.

        Args:
            arm: Selected arm
            context: Context vector (ignored)
            reward: Observed reward (ignored, only tracks frequency)
        """
        self.arm_counts[arm] += 1
        self.total_count += 1
