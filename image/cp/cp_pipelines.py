import os
import pickle
import random
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torchcp

from cp.cp_predictors import MondrianGroupPredictor, SplitPredictor, WeightedSplitPredictor
from cp.cp_utils import get_num_bins, get_score_function, random_test_cal_indices


def get_cp_method(
    method: str,
    model: Any,
    score_function: Any,
    num_classes: int,
    subsample: Optional[bool] = None,
    partitioning: Optional[str] = None,
    save_results: Optional[bool] = None,
    log_dir: Optional[str] = None,
    **kwargs: Any
) -> Any:
    """
    Initialize the appropriate conformal predictor.
    """
    if method == "baseline":
        print("[method] Using SplitPredictor")
        return SplitPredictor(
            score_function=score_function,
            model=model,
            num_classes=num_classes,
            subsample=subsample,
            target_partition=partitioning,
            log_dir=log_dir
        )
    elif method == "mondrian":
        print("[method] Using MondrianGroupPredictor")
        return MondrianGroupPredictor(
            score_function=score_function,
            model=model,
            num_classes=num_classes,
            subsample=subsample,
            target_partition=partitioning,
            log_dir=log_dir,
            **kwargs
        )
    elif method == "weighted":
        print("[method] Using WeightedSplitPredictor")
        return WeightedSplitPredictor(
            score_function=score_function,
            model=model,
            num_classes=num_classes,
            **kwargs
        )


def run_calibration_trials(
    model: Any,
    alpha: float,
    data_module: Any,
    num_resamples: int = 10,
    num_classes: int = 10,
    score_fn: str = "APS",
) -> Dict[str, tuple[float, float]]:
    """
    Runs split conformal prediction (CP) over multiple resampling trials,
    averaging coverage and set size results.

    Args:
        model (Any): Trained classification model.
        alpha (float): Desired miscoverage level (e.g., 0.1 for 90% coverage).
        data_module (Any): Data module providing test dataloader.
        num_resamples (int): Number of random splits to average over.
        num_classes (int): Number of output classes.
        score_fn (str): Conformal score function to use ("APS" or "THR").

    Returns:
        Dict[str, tuple[float, float]]: Dictionary with mean and std of coverage and set size.
    """


    # Select score function (e.g. APS or THR)
    if score_fn in ["THR", "APS"]:
        score_function = getattr(torchcp.classification.score, score_fn)()
    else:
        raise ValueError("Only THR and APS currently implemented as score functions.")

    print(f"Using {score_fn} for scoring...")

    # Initialize split conformal predictor
    cp_method = SplitPredictor(
        score_function=score_function,
        model=model,
        num_classes=num_classes,
        subsample=False,
        target_partition='none'
    )

    # Load test data
    test_loader = data_module.test_dataloader()

    # Collect scores from test data (used for splitting later)
    cp_method.collect_scores(test_loader)

    # Determine dataset size
    if hasattr(cp_method, "num_samples"):
        data_size = cp_method.num_samples
    else:
        data_size = len(test_loader.dataset)

    print(f"{data_size} samples used for testing and calibration.")

    coverages = []
    widths = []

    # Repeat conformal evaluation over random splits
    for _ in range(num_resamples):
        # Randomly split into calibration and test indices
        calibration_indices, test_indices = random_test_cal_indices(data_size=data_size)

        # Evaluate conformal predictor on this split
        trial_results = cp_method.evaluate(
            cal_indices=calibration_indices,
            test_indices=test_indices,
            alpha=alpha
        )

        # Record coverage and set size
        cov = trial_results["Coverage_rate"]
        width = trial_results["Average_size"]

        coverages.append(cov * 100)
        widths.append(width)

    # Compute mean and std of results
    results = {
        "coverage": (
            np.round(np.mean(coverages), decimals=2),
            np.round(np.std(coverages), decimals=3)
        ),
        "width": (
            np.round(np.mean(widths), decimals=3),
            np.round(np.std(widths), decimals=3)
        ),
    }

    print("--------- Average results ----------")
    print(f"average coverage: {results['coverage'][0]} ± {results['coverage'][1]}")
    print(f"average set size: {results['width'][0]} ± {results['width'][1]}")

    return results



def run_mcp_trials(
    model: Any,
    alpha: float,
    data_module: Any,
    num_resamples: int = 10,
    num_classes: int = 10,
    method: str = "baseline",
    score_fn: str = "APS",
    print_class_results: bool = False,
    print_partition_results: bool = False,
    subsample: bool = True,
    partitioning: str = "none",
    num_partitions: int = 8,
    save_results: bool = False,
    log_dir: Optional[str] = None,
    **kwargs: Any
) -> Dict[str, Tuple[float, float]]:

    print(kwargs)

    score_function = get_score_function(score_fn)

    cp_method = get_cp_method(
        method=method,
        model=model,
        score_function=score_function,
        num_classes=num_classes,
        subsample=subsample,
        partitioning=partitioning,
        save_results=save_results,
        log_dir=log_dir,
        **kwargs
    )

    test_loader = data_module.test_dataloader()

    cp_method.collect_scores(test_loader)

    if hasattr(cp_method, "num_samples"):
        data_size = cp_method.num_samples
    else:
        data_size = len(test_loader.dataset)

    print(f"{data_size} samples used for testing and calibration.")
    coverages = []
    widths = []

    class_widths = {i: [] for i in range(num_classes)}
    class_coverages = {i: [] for i in range(num_classes)}

    partition_widths = {i: [] for i in range(num_partitions)}
    partition_coverages = {i: [] for i in range(num_partitions)}

    num_target_partitions = get_num_bins(partitioning, num_classes)

    target_partition_coverages = {i: [] for i in range(num_target_partitions)}

    for _ in range(num_resamples):
        
        # resample test/cal indices
        calibration_indices, test_indices = random_test_cal_indices(data_size=data_size)

        trial_results = cp_method.evaluate(
            cal_indices=calibration_indices, 
            test_indices=test_indices, 
            alpha=alpha
        )
        
        # gather trial results
        cov = trial_results["Coverage_rate"]
        width = trial_results["Average_size"]
        
        coverages.append(cov * 100)
        widths.append(width)

        if print_class_results:
            class_cov = trial_results["Class_coverage"]
            class_width = trial_results["Class_set_size"]

            for i in class_cov.keys():
                class_coverages[i].append(class_cov[i] * 100)
                class_widths[i].append(class_width[i])

        if "Target_partition_coverage" in trial_results:
            target_partition_cov = trial_results["Target_partition_coverage"]
            for i in target_partition_cov.keys():
                target_partition_coverages[i].append(target_partition_cov[i] * 100)

        if print_partition_results:
            partition_cov = trial_results["Partition_coverage"]
            partition_width = trial_results["Partition_set_size"]

            for i in partition_cov.keys():
                partition_coverages[i].append(partition_cov[i] * 100)
                partition_widths[i].append(partition_width[i])

    if save_results:
        if method == "baseline":
            stored_results = {
                "coverage": coverages,
                "width": widths,
                "class_coverage": class_coverages,
                "class_width": class_widths
            }
        else:
            stored_results = {
                "coverage": coverages,
                "width": widths,
                "class_coverage": class_coverages,
                "class_width": class_widths,
                "partition_coverage": partition_coverages,
                "partition_width": partition_widths
            }

        if "Target_partition_coverage" in trial_results:
            stored_results["target_partition_coverage"] = target_partition_coverages
        
        
        with open(f'{log_dir}{method}.pkl', 'wb') as handle:
            pickle.dump(stored_results, handle)

    # gather global results
    results = {
        "coverage": (np.round(np.mean(coverages), decimals=2), np.round(np.std(coverages), decimals=3)),
        "width": (np.round(np.mean(widths), decimals=3), np.round(np.std(widths),  decimals=3)),
    }

    if print_class_results:
        for i in range(num_classes):
            results[f"coverage_c{i}"] = (np.round(np.mean(class_coverages[i]), decimals=2), np.round(np.std(class_coverages[i]), decimals=3))
            results[f"set_size_c{i}"] = (np.round((np.mean(class_widths[i])), decimals=3), np.round(np.std(class_widths[i]), decimals=3))

        print("--------- Class results ----------")
        print("Coverage:")
        for i in range(num_classes):
            print(f"coverage_c{i}: {results[f'coverage_c{i}'][0]} +- {results[f'coverage_c{i}'][1]}")

        print("Set size:")
        for i in range(num_classes):        
            print(f"set_size_c{i}: {results[f'set_size_c{i}'][0]} +- {results[f'set_size_c{i}'][1]}")
    
    if "Target_partition_coverage" in trial_results:
        for i in range(num_target_partitions):
            results[f"coverage_t_p{i}"] = (np.round(np.mean(target_partition_coverages[i]), decimals=2), np.round(np.std(target_partition_coverages[i]), decimals=3))
        
        print("--------- Target partition results ----------")
        print("Coverage:")
        for i in range(num_target_partitions):
            print(f"coverage_t_p{i}: {results[f'coverage_t_p{i}'][0]} +- {results[f'coverage_t_p{i}'][1]}")

    if print_partition_results:
        for i in range(num_partitions):
            results[f"coverage_p{i}"] = (np.round(np.mean(partition_coverages[i]), decimals=2), np.round(np.std(partition_coverages[i]), decimals=3))
            results[f"set_size_p{i}"] = (np.round((np.mean(partition_widths[i])), decimals=3), np.round(np.std(partition_widths[i]), decimals=3))

        print("--------- Partition results ----------")
        print("Coverage:")
        for i in range(num_partitions):
            print(f"coverage_p{i}: {results[f'coverage_p{i}'][0]} +- {results[f'coverage_p{i}'][1]}")

        print("Set size:")
        for i in range(num_partitions):        
            print(f"set_size_p{i}: {results[f'set_size_p{i}'][0]} +- {results[f'set_size_p{i}'][1]}")

    print("--------- Average results ----------")
    print(f"average coverage: {results['coverage'][0]} +- {results['coverage'][1]}")
    print(f"average set size: {results['width'][0]} +- {results['width'][1]}")

    return results

def run_wcp_trials(
    model, 
    alpha, 
    data_module, 
    num_resamples=10, 
    num_classes=10, 
    method="baseline", 
    score_fn="APS", 
    shifted_data_module=None, 
    print_class_results=False,
    print_partition_results=False,
    subsample=False,
    partitioning="none",
    save_results=False,
    **kwargs
):

    print(kwargs)

    score_function = get_score_function(score_fn)

    # if method == "baseline":
        
    #     cp_method = SplitPredictor(
    #         score_function=score_function,
    #         model=model,
    #         num_classes=num_classes,
    #         subsample=subsample,
    #         target_partition=partitioning
    #     )
    # else:
    #     cp_method = WeightedSplitPredictor(
    #         score_function=score_function,
    #         model=model,
    #         num_classes=num_classes,
    #         **kwargs
        # )

    cp_method = get_cp_method(
        method=method,
        model=model,
        score_function=score_function,
        num_classes=num_classes,
        **kwargs
    )


    if shifted_data_module is not None:
        shifted_loader = shifted_data_module.test_dataloader()
    else:
        shifted_loader = None


    test_loader = data_module.test_dataloader()

    cp_method.collect_scores(test_loader, shifted_loader)


    if hasattr(cp_method, "num_samples"):
        data_size = cp_method.num_samples
    else:
        data_size = len(test_loader.dataset)

    print(f"{data_size} samples used for testing and calibration.")
    coverages = []
    widths = []

    class_widths = {i: [] for i in range(num_classes)}
    class_coverages = {i: [] for i in range(num_classes)}


    num_partitions = 8
    partition_widths = {i: [] for i in range(num_partitions)}
    partition_coverages = {i: [] for i in range(num_partitions)}

    num_target_partitions = get_num_bins(partitioning, num_classes)


    target_partition_coverages = {i: [] for i in range(num_target_partitions)}

    for _ in range(num_resamples):
        
        # resample test/cal indices
        num_test = int(data_size * 0.5)
        indices = list(range(data_size))
        random.shuffle(indices)
        test_indices = indices[:num_test]
        calibration_indices = indices[num_test:]

        trial_results = cp_method.evaluate(cal_indices=calibration_indices, test_indices=test_indices, alpha=alpha)
        
        # gather trial results
        cov = trial_results["Coverage_rate"]
        width = trial_results["Average_size"]
        
        coverages.append(cov)
        widths.append(width)

        if print_class_results:
            class_cov = trial_results["Class_coverage"]
            class_width = trial_results["Class_set_size"]

            for i in class_cov.keys():
                class_coverages[i].append(class_cov[i])
                class_widths[i].append(class_width[i])

        if "Target_partition_coverage" in trial_results:
            target_partition_cov = trial_results["Target_partition_coverage"]
            for i in target_partition_cov.keys():
                target_partition_coverages[i].append(target_partition_cov[i])


        if print_partition_results:
            partition_cov = trial_results["Partition_coverage"]
            partition_width = trial_results["Partition_set_size"]

            for i in partition_cov.keys():
                partition_coverages[i].append(partition_cov[i])
                partition_widths[i].append(partition_width[i])

    if save_results:
        if method == "baseline":
            stored_results = {
                "coverage": coverages,
                "width": widths,
                "class_coverage": class_coverages,
                "class_width": class_widths
            }
        else:
            stored_results = {
                "coverage": coverages,
                "width": widths,
                "class_coverage": class_coverages,
                "class_width": class_widths,
                "partition_coverage": partition_coverages,
                "partition_width": partition_widths
            }

        if "Target_partition_coverage" in trial_results:
            stored_results["target_partition_coverage"] = target_partition_coverages
        

        with open(f'./temp_data/mondrian/{partitioning}_{method}_cp_results.pkl', 'wb') as handle:
            pickle.dump(stored_results, handle)


    # gather global results
    results = {
        "coverage": (np.round((np.mean(coverages) * 100), decimals=2), np.round(np.std(coverages), decimals=3)),
        "width": (np.round(np.mean(widths), decimals=3), np.round(np.std(widths),  decimals=3)),
    }


    if print_class_results:
        for i in range(num_classes):
            results[f"coverage_c{i}"] = (np.round((np.mean(class_coverages[i]) * 100), decimals=2), np.round(np.std(class_coverages[i]), decimals=3))
            results[f"set_size_c{i}"] = (np.round((np.mean(class_widths[i])), decimals=3), np.round(np.std(class_widths[i]), decimals=3))

        print("--------- Class results ----------")
        print("Coverage:")
        for i in range(num_classes):
            print(f"coverage_c{i}: {results[f'coverage_c{i}'][0]} +- {results[f'coverage_c{i}'][1]}")

        print("Set size:")
        for i in range(num_classes):        
            print(f"set_size_c{i}: {results[f'set_size_c{i}'][0]} +- {results[f'set_size_c{i}'][1]}")
    
    if "Target_partition_coverage" in trial_results:
        for i in range(num_target_partitions):
            results[f"coverage_t_p{i}"] = (np.round((np.mean(target_partition_coverages[i]) * 100), decimals=2), np.round(np.std(target_partition_coverages[i]), decimals=3))
        
        print("--------- Partition results ----------")
        print("Coverage:")
        for i in range(num_target_partitions):
            print(f"coverage_t_p{i}: {results[f'coverage_t_p{i}'][0]} +- {results[f'coverage_t_p{i}'][1]}")

    if print_partition_results:
        for i in range(num_partitions):
            results[f"coverage_p{i}"] = (np.round((np.mean(partition_coverages[i]) * 100), decimals=2), np.round(np.std(partition_coverages[i]), decimals=3))
            results[f"set_size_p{i}"] = (np.round((np.mean(partition_widths[i])), decimals=3), np.round(np.std(partition_widths[i]), decimals=3))

        print("--------- Partition results ----------")
        print("Coverage:")
        for i in range(num_partitions):
            print(f"coverage_p{i}: {results[f'coverage_p{i}'][0]} +- {results[f'coverage_p{i}'][1]}")

        print("Set size:")
        for i in range(num_partitions):        
            print(f"set_size_p{i}: {results[f'set_size_p{i}'][0]} +- {results[f'set_size_p{i}'][1]}")

    print("--------- Average results ----------")
    print(f"average coverage: {results['coverage'][0]} +- {results['coverage'][1]}")
    print(f"average set size: {results['width'][0]} +- {results['width'][1]}")


    return results
