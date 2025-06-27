import random
from typing import Any, List, Tuple

import torchcp

def get_score_function(score_fn: str) -> Any:
    """
    Dynamically fetch a scoring function class from torchcp.
    """
    if score_fn not in ["THR", "APS"]:
        raise ValueError("Only 'THR' and 'APS' are supported score functions.")
    print(f"[score] Using {score_fn}")
    return getattr(torchcp.classification.score, score_fn)()


def random_test_cal_indices(
    data_size: int, 
    proportion_test: float = 0.5
) -> Tuple[List[int], List[int]]:
    """
    Randomly splits indices into test and calibration sets.

    Args:
        data_size (int): Total number of data points.
        proportion_test (float): Proportion of data to assign to the test set (default: 0.5).

    Returns:
        Tuple[List[int], List[int]]: A tuple containing two lists:
            - calibration_indices: Indices not in the test set
            - test_indices: Randomly selected indices for the test set
    """
    num_test = int(data_size * proportion_test)
    
    # Generate all indices and shuffle them
    indices = list(range(data_size))
    random.shuffle(indices)
    
    # Split the shuffled indices
    test_indices = indices[:num_test]
    calibration_indices = indices[num_test:]
    
    return calibration_indices, test_indices