#!/usr/bin/env python
# coding: utf-8




import pandas as pd
from scipy.sparse import lil_matrix
import numpy as np


def train_test_split(ratings, active_users, fraction: float, count: int):
    """
        Args:
            ratings: Sparse user-item matrix.
            active_users: Active users in descending order.
            fraction: Fraction of users whose interactions
              will be considered for test set.
            count: Number of items per user to move to
              test set.
        Returns:
            train_data: Sparse matrix.
            test_data: Sparse matrix.
            active_users: indices of people to whom we
              make recommendations.
    """
    train_set = ratings.copy()
    test_set = lil_matrix(train_set.shape)
    top_active_users = int(fraction*ratings.shape[0])
    active_users = active_users[:top_active_users]
    for user in active_users:
        size = min(len(ratings.getrow(user).indices), count)
        indices = np.random.choice(ratings.getrow(user).indices, size = size, replace = False)
        train_set[user, indices] = 0
        test_set[user, indices] = ratings[user, indices]
    
    return train_set.tocsr(), test_set.tocsr(), active_users