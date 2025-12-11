"""
Feature engineering module for MovieLens 1M dataset.

This module provides functions to encode user and movie features into
numerical vectors suitable for the LinUCB algorithm.
"""

import numpy as np
import pandas as pd
from typing import Dict


def encode_user_features(users_df: pd.DataFrame) -> Dict[int, np.ndarray]:
    """
    Encode user features as one-hot vectors.

    Encoding scheme (30 dimensions total):
    - Gender: 2 dimensions (positions 0-1)
      - M (Male): [1, 0]
      - F (Female): [0, 1]
    - Age: 7 dimensions (positions 2-8)
      - Age groups: 1, 18, 25, 35, 45, 50, 56
      - One-hot encoding for each group
    - Occupation: 21 dimensions (positions 9-29)
      - Occupation codes: 0-20
      - One-hot encoding for each occupation

    Args:
        users_df: DataFrame with user information containing columns:
                  UserID, Gender, Age, Occupation, Zipcode

    Returns:
        Dictionary mapping UserID to feature vector (30 dimensions)
        Each feature vector is a numpy array with binary values (0 or 1)

    Raises:
        KeyError: If required columns are missing from users_df
        ValueError: If Gender, Age, or Occupation values are invalid

    Example:
        >>> users = load_users('datasets/users.dat')
        >>> features = encode_user_features(users)
        >>> len(features)
        6040
        >>> features[1].shape
        (30,)
        >>> features[1].sum()
        3.0
        >>> # Verify feature dimension allocation
        >>> user_1_features = features[1]
        >>> gender_dims = user_1_features[0:2].sum()  # Should be 1
        >>> age_dims = user_1_features[2:9].sum()      # Should be 1
        >>> occ_dims = user_1_features[9:30].sum()     # Should be 1
        >>> int(gender_dims + age_dims + occ_dims)
        3
    """
    # Validate required columns
    required_cols = ['UserID', 'Gender', 'Age', 'Occupation']
    for col in required_cols:
        if col not in users_df.columns:
            raise KeyError(f"Missing required column: {col}")

    user_features = {}

    # Gender mapping
    gender_map = {'M': 0, 'F': 1}

    # Age groups (7 categories) in sorted order
    age_categories = [1, 18, 25, 35, 45, 50, 56]

    # Occupations: 21 categories (0-20)

    for _, row in users_df.iterrows():
        user_id = row['UserID']

        # Initialize feature vector (30 dims)
        features = np.zeros(30, dtype=np.float32)

        # Gender encoding (positions 0-1)
        gender = row['Gender']
        if gender not in gender_map:
            raise ValueError(f"Invalid gender value: {gender}. Expected 'M' or 'F'")
        gender_idx = gender_map[gender]
        features[gender_idx] = 1

        # Age encoding (positions 2-8)
        age = row['Age']
        if age not in age_categories:
            raise ValueError(f"Invalid age value: {age}. Expected one of {age_categories}")
        age_idx = age_categories.index(age)
        features[2 + age_idx] = 1

        # Occupation encoding (positions 9-29)
        occ = row['Occupation']
        if not (0 <= occ <= 20):
            raise ValueError(f"Invalid occupation value: {occ}. Expected 0-20")
        features[9 + occ] = 1

        user_features[user_id] = features

    return user_features


def encode_movie_features(movies_df: pd.DataFrame) -> Dict[int, np.ndarray]:
    """
    Encode movie genres as multi-hot vectors.

    Each of the 18 genres gets one dimension. A movie can have multiple
    genres, so multiple dimensions can be 1 (multi-hot encoding).

    Genre list (18 total):
    Action, Adventure, Animation, Children's, Comedy, Crime, Documentary,
    Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi,
    Thriller, War, Western

    Args:
        movies_df: DataFrame with movie information containing columns:
                   MovieID, Title, Genres

    Returns:
        Dictionary mapping MovieID to feature vector (18 dimensions)
        Each feature vector is a numpy array with binary values (0 or 1)

    Raises:
        KeyError: If required columns are missing from movies_df
        ValueError: If Genres column contains invalid format

    Example:
        >>> movies = load_movies('datasets/movies.dat')
        >>> features = encode_movie_features(movies)
        >>> len(features)
        3883
        >>> features[1].shape
        (18,)
        >>> # Movies can have multiple genres
        >>> features[1].sum() >= 1
        True
        >>> # Example: Action|Adventure movie should have 2 genres
        >>> action_idx = 0
        >>> adventure_idx = 1
        >>> # Verify multi-hot encoding works
        >>> all(0 <= f.sum() <= 18 for f in features.values())
        True
    """
    # Validate required columns
    required_cols = ['MovieID', 'Title', 'Genres']
    for col in required_cols:
        if col not in movies_df.columns:
            raise KeyError(f"Missing required column: {col}")

    # 18 possible genres in MovieLens 1M
    genres_list = [
        "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
        "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]
    genre_to_idx = {genre: i for i, genre in enumerate(genres_list)}

    movie_features = {}

    for _, row in movies_df.iterrows():
        movie_id = row['MovieID']
        genres_str = row['Genres']

        # Initialize feature vector (18 dims)
        features = np.zeros(18, dtype=np.float32)

        # Split genres by pipe separator
        if not isinstance(genres_str, str):
            raise ValueError(f"Invalid Genres format for MovieID {movie_id}: {genres_str}")

        genres = genres_str.split('|')

        # Set 1 for each genre present
        for genre in genres:
            if genre in genre_to_idx:
                features[genre_to_idx[genre]] = 1
            # Note: We silently ignore unknown genres to be robust

        movie_features[movie_id] = features

    return movie_features


def create_context(
    user_features: np.ndarray,
    movie_features: np.ndarray
) -> np.ndarray:
    """
    Combine user and movie features into context vector.

    Creates a single context vector by concatenating user features
    (30 dimensions) and movie features (18 dimensions).

    Args:
        user_features: User feature vector (30 dimensions)
        movie_features: Movie feature vector (18 dimensions)

    Returns:
        Context vector (48 dimensions) = [user_features | movie_features]

    Raises:
        ValueError: If input dimensions are incorrect

    Example:
        >>> user_feat = np.ones(30)
        >>> movie_feat = np.ones(18)
        >>> context = create_context(user_feat, movie_feat)
        >>> context.shape
        (48,)
        >>> context[:30].sum()
        30.0
        >>> context[30:].sum()
        18.0
    """
    # Validate input dimensions
    if user_features.shape != (30,):
        raise ValueError(f"User features must be 30-dimensional, got {user_features.shape}")
    if movie_features.shape != (18,):
        raise ValueError(f"Movie features must be 18-dimensional, got {movie_features.shape}")

    # Concatenate features
    return np.concatenate([user_features, movie_features])


def rating_to_reward(rating: float, threshold: float = 4.0) -> float:
    """
    Convert rating to binary reward.

    Transforms a 1-5 star rating into a binary reward signal:
    - rating >= threshold: reward = 1 (positive)
    - rating < threshold: reward = 0 (negative)

    Default threshold is 4.0, meaning ratings of 4 and 5 are considered
    positive feedback.

    Args:
        rating: User rating value (typically 1-5)
        threshold: Rating threshold for positive reward (default: 4.0)

    Returns:
        Binary reward: 1.0 if rating >= threshold, else 0.0

    Raises:
        ValueError: If rating is not in valid range [1, 5]

    Example:
        >>> rating_to_reward(5)
        1.0
        >>> rating_to_reward(4)
        1.0
        >>> rating_to_reward(3)
        0.0
        >>> rating_to_reward(1)
        0.0
        >>> rating_to_reward(4.5, threshold=4.5)
        1.0
    """
    # Validate rating range
    if not (1 <= rating <= 5):
        raise ValueError(f"Rating must be in range [1, 5], got {rating}")

    return 1.0 if rating >= threshold else 0.0


def batch_create_contexts(
    user_features_dict: Dict[int, np.ndarray],
    movie_features_dict: Dict[int, np.ndarray],
    user_ids: np.ndarray,
    movie_ids: np.ndarray
) -> np.ndarray:
    """
    Create context vectors for multiple user-movie pairs efficiently.

    This is a utility function for batch processing during evaluation.

    Args:
        user_features_dict: Dictionary mapping UserID to user feature vectors
        movie_features_dict: Dictionary mapping MovieID to movie feature vectors
        user_ids: Array of UserIDs
        movie_ids: Array of MovieIDs (must have same length as user_ids)

    Returns:
        Array of context vectors, shape (n_pairs, 48)

    Raises:
        ValueError: If user_ids and movie_ids have different lengths
        KeyError: If any ID is not found in the respective dictionary

    Example:
        >>> user_feats = {1: np.ones(30), 2: np.ones(30)}
        >>> movie_feats = {100: np.ones(18), 200: np.ones(18)}
        >>> user_ids = np.array([1, 2])
        >>> movie_ids = np.array([100, 200])
        >>> contexts = batch_create_contexts(user_feats, movie_feats, user_ids, movie_ids)
        >>> contexts.shape
        (2, 48)
    """
    if len(user_ids) != len(movie_ids):
        raise ValueError(
            f"user_ids and movie_ids must have same length. "
            f"Got {len(user_ids)} and {len(movie_ids)}"
        )

    n_pairs = len(user_ids)
    contexts = np.zeros((n_pairs, 48), dtype=np.float32)

    for i, (user_id, movie_id) in enumerate(zip(user_ids, movie_ids)):
        if user_id not in user_features_dict:
            raise KeyError(f"UserID {user_id} not found in user_features_dict")
        if movie_id not in movie_features_dict:
            raise KeyError(f"MovieID {movie_id} not found in movie_features_dict")

        contexts[i] = create_context(
            user_features_dict[user_id],
            movie_features_dict[movie_id]
        )

    return contexts
