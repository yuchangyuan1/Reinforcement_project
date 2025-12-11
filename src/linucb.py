"""
LinUCB Algorithm Implementation.

This module implements the Linear Upper Confidence Bound (LinUCB) algorithm
for contextual bandit problems. LinUCB is particularly suitable for scenarios
where the expected reward is a linear function of the context features.

The implementation follows:
Li et al. (2010), "A Contextual-Bandit Approach to Personalized News Article
Recommendation", WWW 2010.

Classes:
    LinUCB: Main algorithm class implementing selection and update methods

Example:
    >>> from src.linucb import LinUCB
    >>> bandit = LinUCB(n_arms=100, n_features=48, alpha=1.0)
    >>> arm, ucb = bandit.select_arm(context)
    >>> bandit.update(arm, context, reward)
"""

import numpy as np
from typing import List, Tuple, Optional


class LinUCB:
    """
    Linear Upper Confidence Bound (LinUCB) algorithm for contextual bandits.

    This implementation follows the algorithm described in:
    Li et al. (2010), "A Contextual-Bandit Approach to Personalized News
    Article Recommendation", WWW 2010.

    The algorithm assumes the expected reward is a linear function of context:
        E[r|x] = theta^T * x

    It uses Upper Confidence Bound (UCB) to balance exploration/exploitation:
        UCB(a) = theta_a^T * x + alpha * sqrt(x^T * A_a^(-1) * x)

    Attributes:
        n_arms (int): Number of arms (movies)
        n_features (int): Dimensionality of context features
        alpha (float): Exploration parameter (controls exploration vs exploitation)
        A (np.ndarray): Design matrix for each arm, shape (n_arms, n_features, n_features)
        b (np.ndarray): Reward vector for each arm, shape (n_arms, n_features)
    """

    def __init__(
        self,
        n_arms: int,
        n_features: int,
        alpha: float = 1.0
    ):
        """
        Initialize LinUCB algorithm.

        Args:
            n_arms: Number of arms (movies in our case)
            n_features: Dimension of context feature vectors
            alpha: Exploration parameter (higher = more exploration)

        Raises:
            ValueError: If n_arms or n_features <= 0, or alpha < 0
        """
        if n_arms <= 0:
            raise ValueError(f"n_arms must be positive, got {n_arms}")
        if n_features <= 0:
            raise ValueError(f"n_features must be positive, got {n_features}")
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")

        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha

        # Initialize A matrices as identity matrices for each arm
        # A_a = I (d x d identity matrix)
        self.A = np.array([np.identity(n_features, dtype=np.float64) for _ in range(n_arms)])

        # Initialize b vectors as zero vectors for each arm
        # b_a = 0 (d-dimensional zero vector)
        self.b = np.zeros((n_arms, n_features), dtype=np.float64)

    def select_arm(
        self,
        context: np.ndarray,
        candidate_arms: Optional[List[int]] = None
    ) -> Tuple[int, float]:
        """
        Select the arm with highest UCB value.

        For each candidate arm a, computes:
            UCB(a) = theta_a^T * x + alpha * sqrt(x^T * A_a^(-1) * x)

        Where:
            - theta_a = A_a^(-1) * b_a (parameter estimate)
            - x is the context vector
            - alpha is the exploration parameter

        Args:
            context: Context feature vector, shape (n_features,)
            candidate_arms: List of available arms. If None, all arms are considered.

        Returns:
            Tuple of (selected_arm_id, ucb_value)

        Raises:
            ValueError: If context has wrong shape or candidate_arms contains invalid indices
        """
        if context.shape != (self.n_features,):
            raise ValueError(
                f"Context must have shape ({self.n_features},), got {context.shape}"
            )

        if candidate_arms is None:
            candidate_arms = list(range(self.n_arms))

        if len(candidate_arms) == 0:
            raise ValueError("candidate_arms cannot be empty")

        # Validate candidate arms
        for arm in candidate_arms:
            if not (0 <= arm < self.n_arms):
                raise ValueError(f"Invalid arm index {arm}, must be in [0, {self.n_arms})")

        best_arm = None
        best_ucb = -float('inf')

        for arm in candidate_arms:
            # Calculate UCB for this arm
            # theta_a = A_a^(-1) * b_a
            theta = np.linalg.solve(self.A[arm], self.b[arm])

            # Predicted reward: theta^T * x
            p = np.dot(theta, context)

            # Confidence bound: alpha * sqrt(x^T * A_a^(-1) * x)
            # Compute A_a^(-1) * x efficiently using solve
            A_inv_x = np.linalg.solve(self.A[arm], context)
            confidence = self.alpha * np.sqrt(np.dot(context, A_inv_x))

            # UCB = predicted reward + confidence bound
            ucb = p + confidence

            if ucb > best_ucb:
                best_ucb = ucb
                best_arm = arm

        return best_arm, float(best_ucb)

    def update(
        self,
        arm: int,
        context: np.ndarray,
        reward: float
    ) -> None:
        """
        Update algorithm parameters after observing reward.

        Updates the A matrix and b vector for the selected arm using
        incremental ridge regression formulas:
            A_a = A_a + x * x^T
            b_a = b_a + r * x

        Args:
            arm: ID of the selected arm
            context: Context feature vector, shape (n_features,)
            reward: Observed reward (typically 0 or 1 for binary rewards)

        Raises:
            ValueError: If arm index is invalid, context has wrong shape,
                       or reward is not a valid number
        """
        if not (0 <= arm < self.n_arms):
            raise ValueError(f"Invalid arm index {arm}, must be in [0, {self.n_arms})")

        if context.shape != (self.n_features,):
            raise ValueError(
                f"Context must have shape ({self.n_features},), got {context.shape}"
            )

        if not np.isfinite(reward):
            raise ValueError(f"Reward must be a finite number, got {reward}")

        # Update A: A_a = A_a + x * x^T
        self.A[arm] += np.outer(context, context)

        # Update b: b_a = b_a + r * x
        self.b[arm] += reward * context

    def get_theta(self, arm: int) -> np.ndarray:
        """
        Get parameter estimate for a specific arm.

        Computes theta_a = A_a^(-1) * b_a using numerically stable solve.

        Args:
            arm: Arm ID

        Returns:
            Parameter estimate vector, shape (n_features,)

        Raises:
            ValueError: If arm index is invalid
        """
        if not (0 <= arm < self.n_arms):
            raise ValueError(f"Invalid arm index {arm}, must be in [0, {self.n_arms})")

        # Use solve instead of inv for numerical stability
        theta = np.linalg.solve(self.A[arm], self.b[arm])
        return theta

    def get_arm_count(self) -> np.ndarray:
        """
        Get the number of times each arm has been updated.

        Approximated by the trace of (A_a - I), since each update adds
        x * x^T to A_a, and trace(x * x^T) = ||x||^2.

        Returns:
            Array of shape (n_arms,) with approximate update counts
        """
        counts = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            # Trace(A_a) - n_features gives a rough estimate
            # More updates -> higher trace
            counts[arm] = np.trace(self.A[arm]) - self.n_features
        return counts

    def reset(self) -> None:
        """
        Reset the algorithm to initial state.

        Resets A matrices to identity and b vectors to zeros.
        """
        self.A = np.array(
            [np.identity(self.n_features, dtype=np.float64) for _ in range(self.n_arms)]
        )
        self.b = np.zeros((self.n_arms, self.n_features), dtype=np.float64)


class HybridLinUCB:
    """
    Hybrid Linear Upper Confidence Bound (LinUCB) algorithm.

    This implementation follows the hybrid model described in:
    Li et al. (2010), "A Contextual-Bandit Approach to Personalized News
    Article Recommendation", WWW 2010, Section 3.2 and Algorithm 1.

    The hybrid model decomposes the reward into shared features (user) and
    arm-specific features (item/movie):
        r_{t,a} = z_t^T * beta* + x_{t,a}^T * theta_a* + epsilon_t

    Where:
        - z_t: Shared feature vector (user features, k-dimensional)
        - x_{t,a}: Arm feature vector (item features, d-dimensional)
        - beta*: Global shared parameters (k-dimensional)
        - theta_a*: Arm-specific parameters (d-dimensional)

    Key advantage: Every update improves global parameters, enabling better
    cold-start performance and cross-arm learning.

    Attributes:
        n_arms (int): Number of arms (movies)
        shared_dim (int): Dimensionality of shared features (k)
        arm_dim (int): Dimensionality of arm features (d)
        alpha (float): Exploration parameter
        A0 (np.ndarray): Global design matrix, shape (k, k)
        b0 (np.ndarray): Global reward vector, shape (k,)
        A (np.ndarray): Per-arm design matrix, shape (n_arms, d, d)
        B (np.ndarray): Cross-feature matrix, shape (n_arms, d, k)
        b (np.ndarray): Per-arm reward vector, shape (n_arms, d)
    """

    def __init__(
        self,
        n_arms: int,
        shared_dim: int,
        arm_dim: int,
        alpha: float = 1.0
    ):
        """
        Initialize Hybrid LinUCB algorithm.

        Args:
            n_arms: Number of arms (movies in our case)
            shared_dim: Dimension of shared feature vectors (k, user features)
            arm_dim: Dimension of arm feature vectors (d, movie features)
            alpha: Exploration parameter (higher = more exploration)

        Raises:
            ValueError: If n_arms, shared_dim, arm_dim <= 0, or alpha < 0
        """
        if n_arms <= 0:
            raise ValueError(f"n_arms must be positive, got {n_arms}")
        if shared_dim <= 0:
            raise ValueError(f"shared_dim must be positive, got {shared_dim}")
        if arm_dim <= 0:
            raise ValueError(f"arm_dim must be positive, got {arm_dim}")
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")

        self.n_arms = n_arms
        self.shared_dim = shared_dim  # k
        self.arm_dim = arm_dim  # d
        self.alpha = alpha

        # Global parameters: A_0 (k x k), b_0 (k,)
        self.A0 = np.identity(shared_dim, dtype=np.float64)
        self.b0 = np.zeros(shared_dim, dtype=np.float64)

        # Per-arm parameters:
        # A_a (d x d): arm-specific design matrix
        # B_a (d x k): cross-feature matrix linking arm and shared features
        # b_a (d,): arm-specific reward vector
        self.A = np.array(
            [np.identity(arm_dim, dtype=np.float64) for _ in range(n_arms)]
        )
        self.B = np.zeros((n_arms, arm_dim, shared_dim), dtype=np.float64)
        self.b = np.zeros((n_arms, arm_dim), dtype=np.float64)

    def select_arm(
        self,
        shared_context: np.ndarray,
        arm_contexts: dict,
        candidate_arms: Optional[List[int]] = None
    ) -> Tuple[int, float]:
        """
        Select the arm with highest UCB value.

        Computes UCB using the hybrid model formula (Li et al. 2010, Eq. 4):
            UCB(a) = z_t^T * beta_hat + x_a^T * theta_hat_a + alpha * sqrt(s_{t,a})

        Where the confidence term s_{t,a} includes cross-feature interactions:
            s_{t,a} = z^T * A0^(-1) * z
                    - 2 * z^T * A0^(-1) * B_a^T * A_a^(-1) * x_a
                    + x_a^T * A_a^(-1) * x_a
                    + x_a^T * A_a^(-1) * B_a * A0^(-1) * B_a^T * A_a^(-1) * x_a

        Args:
            shared_context: Shared feature vector z_t, shape (shared_dim,)
            arm_contexts: Dictionary {arm_id: arm_feature_vector}, where each
                         arm_feature_vector has shape (arm_dim,)
            candidate_arms: List of available arms. If None, uses all arms
                           with features in arm_contexts.

        Returns:
            Tuple of (selected_arm_id, ucb_value)

        Raises:
            ValueError: If shared_context has wrong shape or no valid candidates
        """
        if shared_context.shape != (self.shared_dim,):
            raise ValueError(
                f"shared_context must have shape ({self.shared_dim},), "
                f"got {shared_context.shape}"
            )

        if candidate_arms is None:
            candidate_arms = list(arm_contexts.keys())

        if len(candidate_arms) == 0:
            raise ValueError("candidate_arms cannot be empty")

        # Compute global parameter estimate: beta_hat = A0^(-1) * b0
        # Use solve for numerical stability
        beta_hat = np.linalg.solve(self.A0, self.b0)

        # Precompute A0^(-1) * z for efficiency
        A0_inv_z = np.linalg.solve(self.A0, shared_context)

        best_arm = None
        best_ucb = -float('inf')

        for arm in candidate_arms:
            arm_context = arm_contexts.get(arm)
            if arm_context is None:
                continue

            if not (0 <= arm < self.n_arms):
                raise ValueError(f"Invalid arm index {arm}, must be in [0, {self.n_arms})")

            if arm_context.shape != (self.arm_dim,):
                raise ValueError(
                    f"arm_context for arm {arm} must have shape ({self.arm_dim},), "
                    f"got {arm_context.shape}"
                )

            # Compute arm parameter estimate: theta_hat_a = A_a^(-1) * (b_a - B_a * beta_hat)
            # Using Li et al. (2010) Equation (3)
            theta_hat = np.linalg.solve(
                self.A[arm],
                self.b[arm] - self.B[arm] @ beta_hat
            )

            # Predicted reward: z^T * beta_hat + x_a^T * theta_hat_a
            pred_reward = np.dot(shared_context, beta_hat) + np.dot(arm_context, theta_hat)

            # Compute confidence term s_{t,a} (Li et al. 2010, Equation 4)
            # Using solve for numerical stability instead of explicit matrix inversion

            # Precompute A_a^(-1) * x_a
            Aa_inv_xa = np.linalg.solve(self.A[arm], arm_context)

            # Term 1: z^T * A0^(-1) * z
            term1 = np.dot(shared_context, A0_inv_z)

            # Term 2: -2 * z^T * A0^(-1) * B_a^T * A_a^(-1) * x_a
            term2 = -2.0 * np.dot(A0_inv_z, self.B[arm].T @ Aa_inv_xa)

            # Term 3: x_a^T * A_a^(-1) * x_a
            term3 = np.dot(arm_context, Aa_inv_xa)

            # Term 4: x_a^T * A_a^(-1) * B_a * A0^(-1) * B_a^T * A_a^(-1) * x_a
            # Compute B_a^T * A_a^(-1) * x_a once, then apply A0^(-1)
            Bt_Aa_inv_xa = self.B[arm].T @ Aa_inv_xa  # shared_dim
            A0_inv_Bt_Aa_inv_xa = np.linalg.solve(self.A0, Bt_Aa_inv_xa)
            term4 = np.dot(Bt_Aa_inv_xa, A0_inv_Bt_Aa_inv_xa)

            s_t_a = term1 + term2 + term3 + term4

            # Ensure non-negative (handle numerical errors)
            s_t_a = max(s_t_a, 0.0)

            # Confidence bound
            confidence = self.alpha * np.sqrt(s_t_a)

            # UCB = predicted reward + confidence bound
            ucb = pred_reward + confidence

            if ucb > best_ucb:
                best_ucb = ucb
                best_arm = arm

        if best_arm is None:
            raise ValueError("No valid arm found in candidates")

        return best_arm, float(best_ucb)

    def update(
        self,
        arm: int,
        shared_context: np.ndarray,
        arm_context: np.ndarray,
        reward: float
    ) -> None:
        """
        Update algorithm parameters after observing reward.

        Implements the update rules from Li et al. (2010) Algorithm 1:
            A_a = A_a + x * x^T
            B_a = B_a + x * z^T
            b_a = b_a + r * x
            A_0 = A_0 + z * z^T  (CRITICAL: global update every time!)
            b_0 = b_0 + r * z

        Key insight: Every update improves global parameters, enabling
        cross-arm learning.

        Args:
            arm: ID of the selected arm
            shared_context: Shared feature vector z_t, shape (shared_dim,)
            arm_context: Arm feature vector x_a, shape (arm_dim,)
            reward: Observed reward (typically 0 or 1 for binary rewards)

        Raises:
            ValueError: If arm index is invalid, contexts have wrong shape,
                       or reward is not a valid number
        """
        if not (0 <= arm < self.n_arms):
            raise ValueError(f"Invalid arm index {arm}, must be in [0, {self.n_arms})")

        if shared_context.shape != (self.shared_dim,):
            raise ValueError(
                f"shared_context must have shape ({self.shared_dim},), "
                f"got {shared_context.shape}"
            )

        if arm_context.shape != (self.arm_dim,):
            raise ValueError(
                f"arm_context must have shape ({self.arm_dim},), "
                f"got {arm_context.shape}"
            )

        if not np.isfinite(reward):
            raise ValueError(f"Reward must be a finite number, got {reward}")

        # Update arm-specific parameters
        # A_a = A_a + x * x^T
        self.A[arm] += np.outer(arm_context, arm_context)

        # B_a = B_a + x * z^T (links arm and shared features)
        self.B[arm] += np.outer(arm_context, shared_context)

        # b_a = b_a + r * x
        self.b[arm] += reward * arm_context

        # Update global parameters (CRITICAL: updated every time!)
        # A_0 = A_0 + z * z^T
        self.A0 += np.outer(shared_context, shared_context)

        # b_0 = b_0 + r * z
        self.b0 += reward * shared_context

    def get_beta(self) -> np.ndarray:
        """
        Get global parameter estimate.

        Computes beta = A_0^(-1) * b_0 using numerically stable solve.

        Returns:
            Global parameter estimate vector, shape (shared_dim,)
        """
        # Use solve instead of inv for numerical stability
        beta = np.linalg.solve(self.A0, self.b0)
        return beta

    def get_theta(self, arm: int) -> np.ndarray:
        """
        Get arm-specific parameter estimate.

        Computes theta_a = A_a^(-1) * (b_a - B_a * beta) using numerically
        stable solve (Li et al. 2010, Equation 3).

        Args:
            arm: Arm ID

        Returns:
            Arm-specific parameter estimate vector, shape (arm_dim,)

        Raises:
            ValueError: If arm index is invalid
        """
        if not (0 <= arm < self.n_arms):
            raise ValueError(f"Invalid arm index {arm}, must be in [0, {self.n_arms})")

        beta = self.get_beta()
        # Use solve instead of inv for numerical stability
        theta = np.linalg.solve(self.A[arm], self.b[arm] - self.B[arm] @ beta)
        return theta

    def reset(self) -> None:
        """
        Reset the algorithm to initial state.

        Resets all matrices and vectors to their initial values.
        """
        self.A0 = np.identity(self.shared_dim, dtype=np.float64)
        self.b0 = np.zeros(self.shared_dim, dtype=np.float64)
        self.A = np.array(
            [np.identity(self.arm_dim, dtype=np.float64) for _ in range(self.n_arms)]
        )
        self.B = np.zeros((self.n_arms, self.arm_dim, self.shared_dim), dtype=np.float64)
        self.b = np.zeros((self.n_arms, self.arm_dim), dtype=np.float64)
