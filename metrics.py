import numpy as np


def calculate_mae(true_od, est_od):
    """
    Mean Absolute Error between true OD and estimated OD.
    """
    return np.mean(np.abs(true_od - est_od))


def calculate_rmse(true_od, est_od):
    """
    Root Mean Squared Error between true OD and estimated OD.
    """
    return np.sqrt(np.mean((true_od - est_od) ** 2))


def calculate_relative_total_error(true_od, est_od):
    """
    Relative error between total real demand and total estimated demand.
    """
    true_total = np.sum(true_od)
    est_total = np.sum(est_od)

    if true_total == 0:
        return 0

    return abs(true_total - est_total) / true_total


def reproduce_link_counts(P, est_od_vector):
    """
    Reconstruct link counts using estimated OD vector.
    """
    return P @ est_od_vector


def calculate_link_count_error(observed_counts, reproduced_counts):
    """
    Difference between observed link counts and reproduced link counts.
    """
    mae = np.mean(np.abs(observed_counts - reproduced_counts))
    rmse = np.sqrt(np.mean((observed_counts - reproduced_counts) ** 2))

    return mae, rmse

