"""
Data loading module for MovieLens 1M dataset.

This module provides functions to load users, movies, and ratings data from
the MovieLens 1M dataset .dat files, and utilities for ID mapping.
"""

import pandas as pd
from typing import Dict, Tuple


def load_users(filepath: str) -> pd.DataFrame:
    """
    Load user data from MovieLens .dat file.

    Args:
        filepath: Path to users.dat file

    Returns:
        DataFrame with columns: UserID, Gender, Age, Occupation, Zipcode
        - UserID: User identifier (int)
        - Gender: 'M' or 'F'
        - Age: Age group code (1, 18, 25, 35, 45, 50, 56)
        - Occupation: Occupation code (0-20)
        - Zipcode: ZIP code (string)

    Raises:
        FileNotFoundError: If the specified file does not exist
        pd.errors.ParserError: If the file format is invalid

    Example:
        >>> users = load_users('datasets/users.dat')
        >>> users.shape
        (6040, 5)
        >>> users.columns.tolist()
        ['UserID', 'Gender', 'Age', 'Occupation', 'Zipcode']
    """
    users = pd.read_csv(
        filepath,
        sep='::',
        engine='python',
        header=None,
        names=['UserID', 'Gender', 'Age', 'Occupation', 'Zipcode'],
        encoding='latin-1'
    )
    return users


def load_movies(filepath: str) -> pd.DataFrame:
    """
    Load movie data from MovieLens .dat file.

    Args:
        filepath: Path to movies.dat file

    Returns:
        DataFrame with columns: MovieID, Title, Genres
        - MovieID: Movie identifier (int)
        - Title: Movie title with year (string)
        - Genres: Pipe-separated genre list (string)

    Raises:
        FileNotFoundError: If the specified file does not exist
        pd.errors.ParserError: If the file format is invalid

    Example:
        >>> movies = load_movies('datasets/movies.dat')
        >>> movies.shape
        (3883, 3)
        >>> 'Action|Adventure' in movies['Genres'].values
        True
    """
    movies = pd.read_csv(
        filepath,
        sep='::',
        engine='python',
        header=None,
        names=['MovieID', 'Title', 'Genres'],
        encoding='latin-1'
    )
    return movies


def load_ratings(filepath: str, sort_by_time: bool = True) -> pd.DataFrame:
    """
    Load rating data from MovieLens .dat file.

    Args:
        filepath: Path to ratings.dat file
        sort_by_time: Whether to sort by timestamp in ascending order.
                      Required for offline evaluation (default: True)

    Returns:
        DataFrame with columns: UserID, MovieID, Rating, Timestamp
        - UserID: User identifier (int)
        - MovieID: Movie identifier (int)
        - Rating: Rating value 1-5 (int)
        - Timestamp: Unix timestamp (int)
        If sort_by_time=True, sorted by Timestamp in ascending order

    Raises:
        FileNotFoundError: If the specified file does not exist
        pd.errors.ParserError: If the file format is invalid

    Example:
        >>> ratings = load_ratings('datasets/ratings.dat')
        >>> ratings.shape
        (1000209, 4)
        >>> ratings['Rating'].min(), ratings['Rating'].max()
        (1, 5)
        >>> # Verify temporal ordering
        >>> all(ratings['Timestamp'].diff()[1:] >= 0)
        True
    """
    ratings = pd.read_csv(
        filepath,
        sep='::',
        engine='python',
        header=None,
        names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
        encoding='latin-1'
    )

    if sort_by_time:
        ratings = ratings.sort_values('Timestamp').reset_index(drop=True)

    return ratings


def create_movie_id_mapping(movies_df: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Create bidirectional mappings between MovieID and array indices.

    MovieIDs in the dataset are not continuous (e.g., 1, 2, 5, 8...).
    This function creates mappings to/from continuous array indices (0, 1, 2, 3...).

    Args:
        movies_df: DataFrame with MovieID column (from load_movies)

    Returns:
        Tuple of (movie_id_to_idx, movie_idx_to_id) where:
        - movie_id_to_idx: Dict mapping original MovieID to array index (0-based)
        - movie_idx_to_id: Dict mapping array index to original MovieID

    Example:
        >>> movies = load_movies('datasets/movies.dat')
        >>> id_to_idx, idx_to_id = create_movie_id_mapping(movies)
        >>> len(id_to_idx) == len(idx_to_id) == len(movies)
        True
        >>> # Verify bidirectional mapping
        >>> all(idx_to_id[id_to_idx[movie_id]] == movie_id
        ...     for movie_id in movies['MovieID'])
        True
        >>> # Indices should be continuous 0 to N-1
        >>> sorted(idx_to_id.keys()) == list(range(len(movies)))
        True
    """
    movie_ids = sorted(movies_df['MovieID'].unique())

    # Create bidirectional mappings
    movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
    movie_idx_to_id = {idx: movie_id for movie_id, idx in movie_id_to_idx.items()}

    return movie_id_to_idx, movie_idx_to_id


def create_user_id_mapping(users_df: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Create bidirectional mappings between UserID and array indices.

    Similar to create_movie_id_mapping, this creates mappings for user IDs.

    Args:
        users_df: DataFrame with UserID column (from load_users)

    Returns:
        Tuple of (user_id_to_idx, user_idx_to_id) where:
        - user_id_to_idx: Dict mapping original UserID to array index (0-based)
        - user_idx_to_id: Dict mapping array index to original UserID

    Example:
        >>> users = load_users('datasets/users.dat')
        >>> id_to_idx, idx_to_id = create_user_id_mapping(users)
        >>> len(id_to_idx) == len(idx_to_id) == len(users)
        True
    """
    user_ids = sorted(users_df['UserID'].unique())

    # Create bidirectional mappings
    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    user_idx_to_id = {idx: user_id for user_id, idx in user_id_to_idx.items()}

    return user_id_to_idx, user_idx_to_id
