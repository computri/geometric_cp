import random
from typing import Any, List, Optional, Tuple, Dict

import torchcp


PARTITION_TO_NUM_BINS = {
    "entropy": 6,
    "edge": 8,
    "color": 8,
}

def get_score_function(score_fn: str) -> Any:
    """
    Dynamically fetch a scoring function class from torchcp.
    """
    if score_fn not in ["THR", "APS"]:
        raise ValueError("Only 'THR' and 'APS' are supported score functions.")
    print(f"[score] Using {score_fn}")
    return getattr(torchcp.classification.score, score_fn)()


def save_distances(
    model: Any,
    data_module: Any,
    num_classes: int = 10,
    score_fn: str = "APS",
    shifted_data_module: Optional[Any] = None,
    **kwargs: Any,
) -> None:

    print(kwargs)

    if score_fn in ["THR", "APS"]:
        score_function = getattr(torchcp.classification.score, score_fn)()
    else:
        raise ValueError("Only THR and APS currently implemented as score functions.")

    cp_method = WeightedSplitPredictor(
        score_function=score_function,
        model=model,
        num_classes=num_classes,
        **kwargs
    )

    test_loader = data_module.test_dataloader()

    if shifted_data_module is not None:
        shifted_loader = shifted_data_module.test_dataloader()
    else:
        shifted_loader = None

    test_loader = data_module.test_dataloader()

    cp_method.collect_scores(test_loader, shifted_loader)

    # resample test/cal indices
    num_test = int(len(test_loader.dataset) * 0.5)
    indices = list(range(len(test_loader.dataset)))
    random.shuffle(indices)
    test_indices = indices[:num_test]
    calibration_indices = indices[num_test:]

    cp_method.save_distances(cal_indices=calibration_indices, test_indices=test_indices)


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


def get_partition_mapping(num_partitions: int) -> Dict[int, int]:
    """
    Returns a mapping from canonical angle values to integer indices,
    depending on the number of partitions.

    Args:
        num_partitions: Must be 4 or 8.

    Returns:
        Dictionary mapping angles to partition indices.

    Raises:
        ValueError: If an unsupported number of partitions is provided.
    """
    angles = {
        4: [0, 90, 180, 270],
        8: [0, 45, 90, 135, 180, 225, 270, 315]
    }

    if num_partitions not in angles:
        raise ValueError(f"Unsupported number of partitions: {num_partitions}. "
                         f"Supported values are {list(angles.keys())}.")

    return {angle: idx for idx, angle in enumerate(angles[num_partitions])}

def get_num_bins(target_partition: str, num_classes: int) -> int:
    if target_partition == "label" or target_partition == "none":
        return num_classes
    elif target_partition in PARTITION_TO_NUM_BINS:
        return PARTITION_TO_NUM_BINS[target_partition]
    else:
        raise ValueError(f"Unknown target_partition: '{target_partition}'")
