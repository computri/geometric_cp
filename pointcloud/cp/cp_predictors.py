# Adapted from: https://github.com/ml-stat-Sustech/TorchCP/blob/master/torchcp/classification/predictor/split.py
# Copyright (C) 2022 SUSTech Machine Learning and Statistics Group
# Licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0)
# See https://www.gnu.org/licenses/lgpl-3.0.html for license details.

import numpy as np
import torch

from torchcp.classification.predictor.base import BasePredictor
from torchcp.utils.common import calculate_conformal_value


class SplitPredictor(BasePredictor):
    """
    Split Conformal Prediction (Vovk et a., 2005).
    Book: https://link.springer.com/book/10.1007/978-3-031-06649-8.
    
    :param score_function: non-conformity score function.
    :param model: a pytorch model.
    :param temperature: the temperature of Temperature Scaling.
    """
    def __init__(
        self, 
        score_function, 
        model=None, 
        temperature=1,
        num_classes=10
    ):
        super().__init__(score_function, model, temperature)

        self.num_classes = num_classes


    def collect_scores(self, dataloader):
        """ Collect scores over the entire cal and test sets. To be subsampled later for cal/test splits.
        In the case of covariate shift between cal and test data, an (optional) additional dataloader can be 
        passed that contains shifted data.
        In this case, the shifted data will be used as the test data.
        Note: the regular and shifted data should have the same sample ordering!

        """

        print("Collecting scores....")
        self._model.eval()
        self.all_cal_scores, self.all_cal_labels = self._collect_scores(dataloader)

        self.all_test_scores, self.all_test_labels = self.all_cal_scores, self.all_cal_labels
        
        print("Score collection done.")

    def _collect_scores(self, dataloader):
        """ Collect scores over the entire cal and test sets. To be subsampled later for cal/test splits
        """
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for examples in dataloader:
                batch_inputs, batch_labels = examples[0].to(self._device), examples[1].to(self._device)

                batch_logits = self._logits_transformation(self._model(batch_inputs)).detach()
                
                logits_list.append(batch_logits)
                labels_list.append(batch_labels)

            logits = torch.cat(logits_list).float().to(self._device)
            
        labels = torch.cat(labels_list).to(self._device).squeeze()
            
        scores = self.score_function(logits)

        return scores, labels


    #############################
    # The calibration process
    # ############################

    def calibrate(self, cal_scores, alpha):
        """ Returns quantile over cal_scores for given alpha"""
        # calculate new quantile with weights
        return calculate_conformal_value(cal_scores, alpha=alpha)
        

    #############################  
    # The prediction process
    ############################
    
    def predict(self, test_scores, q_hat):
        """Get prediction sets for entire batch of test scores given a quantile value"""
        return self._generate_prediction_set(test_scores, q_hat)


    def evaluate(self, cal_indices, test_indices, alpha):
        """
        Args:
            cal_indices: List: indices that define the calibration split
            test_indices: List: indices that define the test split
            alpha: error rate
        """

        # Collect cal and test splits
        cal_scores, cal_labels = self.all_cal_scores[cal_indices], self.all_cal_labels[cal_indices]
        test_scores, test_labels = self.all_test_scores[test_indices], self.all_test_labels[test_indices]

        #select only scores from labels
        cal_scores = cal_scores[torch.arange(cal_scores.shape[0], device=cal_scores.device), cal_labels] 
        

        # for accumulating the class-wise statistics
        class_set_sizes = {
            i: [] for i in range(self.num_classes)
        }
        

        q_hat = self.calibrate(cal_scores, alpha)

        prediction_sets = self.predict(test_scores, q_hat)


        for i, label in enumerate(test_labels):
            class_set_sizes[label.item()].append(prediction_sets[i].sum().item())

        class_sizes = self.get_set_size_per_class(class_set_sizes)
        class_cov = self.get_coverage_per_class(test_labels, prediction_sets)
        
        res_dict = {"Coverage_rate": self._metric('coverage_rate')(prediction_sets, test_labels),
                    "Average_size": self._metric('average_size')(prediction_sets, test_labels),
                    "Class_coverage": class_cov,
                    "Class_set_size": class_sizes}
        
        return res_dict

    def get_set_size_per_class(self, class_set_sizes):
        return {label: np.mean(label_set_sizes) for label, label_set_sizes in class_set_sizes.items()}

    def get_coverage_per_class(self, test_labels, prediction_sets):
        class_coverages = {
            i: [] for i in range(self.num_classes)
        }
        
        for i, label  in enumerate(test_labels):
            
            class_coverages[label.item()].append(prediction_sets[i][label.item()].item())

        return {label: np.mean(class_coverage) for label, class_coverage in class_coverages.items()}

