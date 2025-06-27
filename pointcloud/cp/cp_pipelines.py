from typing import Any, Dict

import numpy as np

from cp.cp_predictors import SplitPredictor
from cp.cp_utils import get_score_function, random_test_cal_indices


def run_calibration_trials(
    model: Any, 
    alpha: float, 
    data_module: Any, 
    num_resamples: int = 10, 
    num_classes: int = 10, 
    score_fn: str ="APS"
) -> Dict[str, tuple[float, float]]:
    score_function = get_score_function(score_fn)
        
    cp_method = SplitPredictor(
        score_function=score_function,
        model=model,
        num_classes=num_classes
    )

    test_loader = data_module.test_dataloader()

    cp_method.collect_scores(test_loader)

    coverages = []
    widths = []

    class_widths = {i: [] for i in range(num_classes)}
    class_coverages = {i: [] for i in range(num_classes)}

    for _ in range(num_resamples):

        calibration_indices, test_indices = random_test_cal_indices(len(test_loader.dataset), proportion_test=0.5)
        

        results = cp_method.evaluate(cal_indices=calibration_indices, test_indices=test_indices, alpha=alpha)
        
        # gather trial results
        cov = results["Coverage_rate"]
        width = results["Average_size"]
        
        coverages.append(cov * 100)
        widths.append(width)

        class_cov = results["Class_coverage"]
        class_width = results["Class_set_size"]

        for i in class_cov.keys():
            class_coverages[i].append(class_cov[i])
            class_widths[i].append(class_width[i])

        
    # gather global results
    results = {
        "coverage": (
            np.round((np.mean(coverages)), decimals=2), 
            np.round(np.std(coverages), decimals=3)
        ),
        "width": (
            np.round(np.mean(widths), decimals=3), 
            np.round(np.std(widths),  decimals=3)
        ),
    }


    for i in range(num_classes):
        results[f"coverage_c{i}"] = (np.round((np.mean(class_coverages[i])), decimals=2), np.round(np.std(class_coverages[i]), decimals=3))
        results[f"set_size_c{i}"] = (np.round((np.mean(class_widths[i])), decimals=3), np.round(np.std(class_widths[i]), decimals=3))

    
    print("--------- Average results ----------")
    print(f"average coverage: {results['coverage'][0]} +- {results['coverage'][1]}")
    print(f"average set size: {results['width'][0]} +- {results['width'][1]}")


    return results



