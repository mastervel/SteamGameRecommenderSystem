from datetime import datetime
import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm

def is_valid_unix_time(value) -> bool:
    # Ensure the value is an integer (or can be safely converted)
    if not isinstance(value, (int, float)):
        return False
    
    # Check if the value is within the valid Unix timestamp range
    try:
        datetime.fromtimestamp(value)
        return True
    except (OverflowError, OSError, ValueError):
        return False

def validate_unix_dict(input_dict):
    return all(is_valid_unix_time(value) for value in input_dict.values())

def find_invalid_entries(input_dict):
    return {k: v for k, v in input_dict.items() if not is_valid_unix_time(v)}

def unix_time_converter(unix_time: int) -> datetime:
    return datetime.fromtimestamp(unix_time).replace(hour=0, minute=0, second=0, microsecond=0)

def time_converter(date_str: str) -> datetime:
    dt = datetime.strptime(date_str, "%d %b, %Y")
    return datetime(dt.year, dt.month, dt.day)

def playtime_normalization(R, matrix_df, time_since_user_dict, epsilon = 1e-6):

    # Create user_time array based on matrix_df.index (users)
    user_time = np.array([time_since_user_dict[str(user)] for user in matrix_df.index])
    
    # Normalize R row-wise by dividing each row by the corresponding user's time
    normalized_matrix = (R / 60) / user_time[:, None]
    
    return np.log(normalized_matrix + epsilon)

import random
from collections import defaultdict

def split_playtime_data(observed_playtime_elements, test_ratio=0.05, seed=42, set_seed=True):
    """
    Splits playtime data into train and test sets.

    Parameters:
    - observed_playtime_elements (list of tuples): List of (user_id, appid) pairs.
    - test_ratio (float): Proportion of each user's data to allocate to the test set (default is 0.05).
    - seed (int): Random seed for reproducibility (default is 42).
    - set_seed (bool): Whether to set the random seed for reproducibility (default is True).

    Returns:
    - train_indices (list of tuples): Training set as (user_id, appid) pairs.
    - test_indices (list of tuples): Test set as (user_id, appid) pairs.
    """
    # Set seed for reproducibility if required
    if set_seed:
        random.seed(seed)

    # Group appids by user_id
    user_playtimes = defaultdict(list)
    for user_id, appid in observed_playtime_elements:
        user_playtimes[user_id].append(appid)

    # Initialize train and test sets
    train_indices = []
    test_indices = []

    # Split data for each user
    for user_id, app_list in user_playtimes.items():
        # Determine the number of playtimes to include in the test set
        test_size = max(1, int(test_ratio * len(app_list)))

        # Randomly select appids for the test set
        test_apps = random.sample(app_list, test_size)

        # Populate train and test sets
        for appid in app_list:
            if appid in test_apps:
                test_indices.append((user_id, appid))
            else:
                train_indices.append((user_id, appid))

    print(f"Train size: {len(train_indices)}, Test size: {len(test_indices)}")
    return train_indices, test_indices

def generate_bias(R, train_indices, user_dict, games_dict):
    
    matrix_indices = [(user_dict[user], games_dict[game]) for user, game in train_indices]
    user_indices, game_indices = np.array(matrix_indices).T
    
    # Generate global mean playtime
    values = R[user_indices, game_indices]
    mean_playtime = np.mean(values)
    
    # Generate user bias vector
    user_sum = defaultdict(float)
    user_count = defaultdict(int)

    for (user, value) in zip(user_indices, values):
        user_sum[user] += value
        user_count[user] += 1

    user_bias = np.zeros(R.shape[0])
    for user in user_sum:
        user_bias[user] = (user_sum[user] / user_count[user]) - mean_playtime
    
    # Generate games bias vector 
    game_sum = defaultdict(float)
    game_count = defaultdict(int)

    for (game, value) in zip(game_indices, values):
        game_sum[game] += value
        game_count[game] += 1

    game_bias = np.zeros(R.shape[1])
    for game in game_sum:
        game_bias[game] = (game_sum[game] / game_count[game]) - mean_playtime

    return mean_playtime, user_bias, game_bias

def apply_svd(R, k=None):
    """
    Apply SVD to matrix R and return the matrices P and Q.
    Optionally, perform dimensionality reduction by keeping the top k singular values.

    Parameters:
    - R: The normalized utility matrix (user-item matrix).
    - k: The number of singular values to keep (optional). If None, all singular values are used.

    Returns:
    - P: User-feature matrix.
    - Q: Item-feature matrix.
    """
    # Step 1: Perform Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(R, full_matrices=False)

    # Step 2: Dimensionality reduction (if k is specified)
    if k is not None:
        U = U[:, :k]
        S = np.diag(S[:k])
        Vt = Vt[:k, :]

    # Step 3: Set P and Q
    P = U  # User-feature matrix
    Q = Vt.T  # Item-feature matrix (transposed V)

    return P, Q

def reconstruct_r(P, Q):
    """
    Reconstruct the matrix R from the user-feature matrix P and item-feature matrix Q.
    
    Parameters:
    - P: User-feature matrix.
    - Q: Item-feature matrix.

    Returns:
    - R_reconstructed: The reconstructed matrix R.
    """
    # Step 1: Reconstruct the matrix R by multiplying P and the transpose of Q
    R_reconstructed = np.dot(P, Q.T)  # or you can use P @ Q.T in Python 3.5+

    return R_reconstructed

def SGD(R, P, Q, train_indices, user_dict, games_dict, n_epochs=100, lmbda=0.1, learning_rate=0.001,
       mu = None, bu = None, bi = None):
    """
    Stochastic Gradient Descent (SGD) for Matrix Factorization with L2 Regularization.
    
    Optimized for efficiency.
    
    Parameters:
    - R: User-item matrix with observed playtimes.
    - train_indices: List of (user_id, appid) pairs representing observed training data.
    - P: User latent factor matrix.
    - Q: Item latent factor matrix.
    - n_epochs: Number of iterations.
    - lmbda: L2 regularization strength.
    - learning_rate: Step size for updates.
    
    Returns:
    - Updated P, Q matrices.
    - List of train losses for each epoch.
    """

    if mu is None:
        mu = np.mean(R[np.nonzero(R)])
    if bu is None:
        bu = np.zeros(R.shape[0])
    if bi is None:
        bi = np.zeros(R.shape[1])

    loss = []

    # Precompute index mappings to avoid repeated lookups
    train_indices = np.array([(user_dict[user_id], games_dict[appid]) for user_id, appid in train_indices])

    for e in tqdm(range(n_epochs), desc="Epochs", unit="epoch"):
        total_loss = 0       
        
        for u, i in train_indices:
            error = R[u, i] - ( mu + bu[u] + bi[i] + np.dot(P[u], Q[i]) )

            # Compute loss efficiently (avoiding redundant norm computations)
            total_loss += error**2 + lmbda * (np.dot(P[u], P[u]) + np.dot(Q[i], Q[i]) + bu[u]**2 + bi[i]**2)

            # Update P and Q using efficient NumPy operations
            P[u] += learning_rate * (error * Q[i] - lmbda * P[u])
            Q[i] += learning_rate * (error * P[u] - lmbda * Q[i])
            bu[u] += learning_rate * (error - lmbda * bu[u])
            bi[i] += learning_rate * (error - lmbda * bi[i])

        loss.append(total_loss / len(train_indices))
    
    return P, Q, bu, bi, loss

def mse(actual_R, estimated_R, test_indices, user_dict, games_dict):

    test_indices = np.array([(user_dict[user_id], games_dict[appid]) for user_id, appid in test_indices])
    rows, cols = test_indices[:, 0], test_indices[:, 1]
    errors = actual_R[rows, cols] - estimated_R[rows, cols]
    
    return np.mean(errors ** 2)

def prediction_matrix(estimated_R, matrix_df, train_indices, test_indices, user_dict, games_dict):
    # Vectorized approach to map user_id, appid to row, col indices for train and test sets
    train_rows = np.array([user_dict[user_id] for user_id, _ in train_indices])
    train_cols = np.array([games_dict[appid] for _, appid in train_indices])
    
    test_rows = np.array([user_dict[user_id] for user_id, _ in test_indices])
    test_cols = np.array([games_dict[appid] for _, appid in test_indices])

    matrix_array = matrix_df.values
    
    # Update the prediction matrix for both training and test indices
    estimated_R[train_rows, train_cols] = matrix_array[train_rows, train_cols]
    estimated_R[test_rows, test_cols] = matrix_array[test_rows, test_cols]

    return estimated_R

def recommend_by_user(user_id, prediction_R, top_n, observed_playtime_elements, appid_to_name, user_dict, games_dict):
    appids = games_dict.keys()

    # Retrieving recommendations on unowned games sorted by top n
    row_index = user_dict[user_id]
    observed_games = set(t[1] for t in observed_playtime_elements if t[0] == user_id)
    predicted_games = set(appids) - observed_games

    playtime_dict = {col: prediction_R[row_index, games_dict[col]] for col in predicted_games}
    sorted_dict = dict(sorted(playtime_dict.items(), key=lambda item: item[1], reverse=True)[:top_n])
    named_dict = {appid_to_name.get(k, k): v for k, v in sorted_dict.items()}

    return named_dict

def get_top_games_by_playtime(observed_playtime_elements, matrix_df, user_dict, games_dict, appid_to_name, top_n=10):
    top_games_by_user = {}

    for user_id in user_dict:
        # Get the observed games for the current user
        row_index = user_dict[user_id]
        observed_games = set(t[1] for t in observed_playtime_elements if t[0] == user_id)

        # Create a dictionary of playtimes for the observed games
        playtime_dict = {col: matrix_df.iloc[row_index, games_dict[col]] for col in observed_games}

        # Sort the playtime dictionary to get the top N games
        sorted_dict = dict(sorted(playtime_dict.items(), key=lambda item: item[1], reverse=True)[:top_n])

        # Add game names to the sorted list
        named_dict = {appid_to_name.get(k, k): v for k, v in sorted_dict.items()}

        # Store the result for the current user
        top_games_by_user[user_id] = named_dict

    return top_games_by_user