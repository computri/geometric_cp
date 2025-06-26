# Adapted from: https://github.com/ml-stat-Sustech/TorchCP/blob/master/torchcp/classification/predictor/split.py
# Copyright (C) 2022 SUSTech Machine Learning and Statistics Group
# Licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0)
# See https://www.gnu.org/licenses/lgpl-3.0.html for license details.


import math
import os
import pickle
import warnings
from functools import partial
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from torchcp.classification.predictor.base import BasePredictor

import cp.weighting_schemes as weighting_schemes
from cp.cp_utils import get_num_bins, get_partition_mapping


def calculate_conformal_value(scores, alpha, default_q_hat=torch.inf):
    """
    Calculate the 1-alpha quantile of scores for conformal prediction.

    This function computes the threshold value (quantile) used to construct prediction sets based on the given
    non-conformity scores and significance level alpha. If the scores are empty or the quantile value exceeds 1,
    it returns the default_q_hat value.

    Args:
        scores (torch.Tensor): Non-conformity scores.
        alpha (float): Significance level, must be between 0 and 1.
        default_q_hat (torch.Tensor or str, optional): Default threshold value to use if scores are empty or invalid.
            If set to "max", it uses the maximum value of scores. Default is torch.inf.

    Returns:
        torch.Tensor: The threshold value used to construct prediction sets.
    
    Raises:
        ValueError: If alpha is not between 0 and 1.
    """
    if default_q_hat == "max":
        default_q_hat = torch.max(scores)
    if alpha >= 1 or alpha <= 0:
        raise ValueError("Significance level 'alpha' must be in [0,1].")
    if len(scores) == 0:
        warnings.warn(
            f"The number of scores is 0, which is a invalid scores. To avoid program crash, the threshold is set as {default_q_hat}.")
        return default_q_hat
    N = scores.shape[0]
    quantile_value = math.ceil((N + 1) * (1 - alpha)) / N
    
    if quantile_value > 1:
        warnings.warn(
            f"The value of quantile exceeds 1. It should be a value in [0,1]. To avoid program crash, the threshold is set as {default_q_hat}.")
        return default_q_hat
    
    return torch.quantile(scores, quantile_value, dim=0, interpolation='linear').to(scores.device)


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
        num_classes=10,
        subsample=False,
        target_partition=None,
        log_dir=None
    ):
        super().__init__(score_function, model, temperature)

        self.num_classes = num_classes
        self.subsample = subsample
        if subsample:
            assert log_dir is not None, "log_dir must be provided for subsampling."

        self.target_partition = target_partition
        
        if log_dir is not None:
            self.log_dir = log_dir 
            os.makedirs(self.log_dir, exist_ok=True)

    def collect_scores(self, dataloader, shifted_dataloader=None):
        """ Collect scores over the entire cal and test sets. To be subsampled later for cal/test splits.
        In the case of covariate shift between cal and test data, an (optional) additional dataloader can be 
        passed that contains shifted data.
        In this case, the shifted data will be used as the test data.
        Note: the regular and shifted data should have the same sample ordering!

        """

        print("Collecting scores....")
        self._model.eval()
        self.all_cal_scores, self.all_cal_labels = self._collect_scores(dataloader)

        if shifted_dataloader is not None:
            print("using shifted")
            self.all_test_scores, self.all_test_labels = self._collect_scores(shifted_dataloader)
            assert torch.all(self.all_cal_labels == self.all_test_labels), "ordering for the two dataloaders should be identical."


        else:
            self.all_test_scores, self.all_test_labels = self.all_cal_scores, self.all_cal_labels

        if self.target_partition is not None:
            self.all_partition_info = self._get_partition_info()
        
        print("Score collection done.")

    def _collect_scores(self, dataloader):
        """ Collect scores over the entire cal and test sets. To be subsampled later for cal/test splits
        """
        logits_list = []
        labels_list = []

        if self.subsample:
            with open(self.log_dir + '/cutoff_indices.pkl', 'rb') as handle:    
                indices = pickle.load(handle)

        with torch.no_grad():
            for i, examples in enumerate(dataloader):

                batch_inputs, batch_labels = examples[0].to(self._device), examples[1].to(self._device)

                if self.subsample:
                    batch_inputs, batch_labels = batch_inputs[indices[i]], batch_labels[indices[i]]

                batch_logits = self._logits_transformation(self._model(batch_inputs)).detach()
                
                logits_list.append(batch_logits)
                labels_list.append(batch_labels)

        logits = torch.cat(logits_list).float().to(self._device)
            
        labels = torch.cat(labels_list).to(self._device)
            
        scores = self.score_function(logits)

        self.num_samples = scores.shape[0]

        return scores, labels

    def _get_partition_info(self):

        if self.subsample:
            with open(self.log_dir + '/cutoff_indices.pkl', 'rb') as handle:    
                indices = pickle.load(handle)
                batch_size = torch.cat(indices, dim=0).max().item() + 1
        partitions = []

        self.num_bins = get_num_bins(self.target_partition, self.num_classes)

        with open(self.log_dir + f'/{self.target_partition}.pkl', 'rb') as handle:
            partition_info = pickle.load(handle)
            start_idx = 0
            i = 0

            partition_info = partition_info.to(indices[0].device)


            while start_idx < partition_info.shape[0]:
                batch = partition_info[start_idx:start_idx + batch_size]
                if self.subsample:
                    batch = batch[indices[i]]

                partitions.append(batch)
                start_idx += batch_size
                i += 1

                
        partitions = torch.cat(partitions).to(self._device)
            
        return partitions

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

        if self.target_partition is not None:
            test_target_partition_info = self.all_partition_info[test_indices]

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
        
        if self.target_partition is not None:
            
            res_dict["Target_partition_coverage"] = self.get_coverage_per_part(test_labels, prediction_sets, test_target_partition_info)

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

    def get_coverage_per_part(self, test_labels, prediction_sets, partition_info):
        part_coverages = {
            i: [] for i in range(self.num_bins)
        }
        for i, label  in enumerate(test_labels):
            part_coverages[partition_info[i].item()].append(prediction_sets[i][label.item()].item())

        return {label: np.mean(p_coverage) for label, p_coverage in part_coverages.items()}

class WeightedSplitPredictor(BasePredictor):
    """
    Split Conformal Prediction (Vovk et a., 2005).
    Book: https://link.springer.com/book/10.1007/978-3-031-06649-8.
    
    :param score_function: non-conformity score function.
    :param model: a pytorch model.
    :param temperature: the temperature of Temperature Scaling.
    """
    def __init__(self, 
        score_function, 
        model=None, 
        temperature=1, 
        similarity_type="inverse_l2",
        w_lambda=1.0,
        w_pow=1.0,
        num_classes=10,
    ):
        super().__init__(score_function, model, temperature)
        
        self.num_classes = num_classes
        self.w_transform = lambda x: 1 + (w_lambda * (x ** w_pow))

        assert similarity_type in ["inverse_l2", "inverse_crossentropy", "inverse_kl", "inverse_hellinger", "inverse_tv", "uniform"], f"Similarity type '{similarity_type}' not recognized."
        self.similarity_type = similarity_type

        self.distance_fn = partial(getattr(weighting_schemes, similarity_type), w_transform=self.w_transform)

        
        
    def collect_scores(self, dataloader, shifted_dataloader=None):
        """ Collect scores over the entire cal and test sets. To be subsampled during eval for cal/test splits.
        In the case of covariate shift between cal and test data, an (optional) additional dataloader can be 
        passed that contains shifted data.
        In this case, the shifted data will be used as the test data.
        Note: the regular and shifted data should have the same sample ordering!

        """
        print("Collecting scores....")
        self._model.eval()
        self.all_cal_scores, self.all_cal_labels, self.all_cal_weight_quantities = self._collect_scores(dataloader)

        if shifted_dataloader is not None:
            print("using shifted")
            self.all_test_scores, self.all_test_labels, self.all_test_weight_quantities = self._collect_scores(shifted_dataloader)
            assert torch.all(self.all_cal_labels == self.all_test_labels), "ordering for the two dataloaders should be identical."
        else:
            self.all_test_scores, self.all_test_labels, self.all_test_weight_quantities = self.all_cal_scores, self.all_cal_labels, self.all_cal_weight_quantities
        
        print("Score collection done.")


    def _collect_scores(self, dataloader):
        logits_list = []
        labels_list = []
        weight_quantities_list = []
        with torch.no_grad():
            for examples in dataloader:
                batch_inputs, batch_labels = examples[0].to(self._device), examples[1].to(self._device)

                weight_quantity = self._get_weight_info(batch_inputs)

                batch_logits = self._logits_transformation(self._model(batch_inputs)).detach()
    
                weight_quantities_list.append(weight_quantity)
                logits_list.append(batch_logits)
                labels_list.append(batch_labels)

            logits = torch.cat(logits_list).float().to(self._device)
            
        labels = torch.cat(labels_list).to(self._device)
            
        weight_quantities = torch.cat(weight_quantities_list).float().to(self._device) # save weight info

        scores = self.score_function(logits)

        return scores, labels, weight_quantities


    # #soft case
    def _get_weight_info(self, inputs):
        """Extracts the quantities that are used in the weight calculation
        Args:
            inputs: Tensor: [N, D_in]: tensor of model inputs

        returns:
            Tensor [N, D_out]: Quantities used to calculate/compare calibration weights
        """
        # weight_quantity = tmp_labels
        group_activ = self._model.canonicalizer.get_group_activations(inputs)
        group_activ = torch.softmax(group_activ, dim=-1)
        return group_activ

        
    
    def _get_weight(self, test_weight_quantity, cal_weight_quantities):

        weights_unnormalized = self.distance_fn(cal_weight_quantities, test_weight_quantity)

        # normalize the weights
        return weights_unnormalized / weights_unnormalized.sum()

    
    def save_distances(self, cal_indices, test_indices):
        cal_scores, cal_labels = self.all_cal_scores[cal_indices], self.all_cal_labels[cal_indices]
        test_scores, test_labels = self.all_test_scores[test_indices], self.all_test_labels[test_indices]

        cal_scores = cal_scores[torch.arange(cal_scores.shape[0], device=cal_scores.device), cal_labels] #select only scores from labels

        cal_weight_quantities = self.all_cal_weight_quantities[cal_indices]

        test_weight_quantities = self.all_test_weight_quantities[test_indices]

        
        weights = []
        for idx, test_score in enumerate(test_scores):
            
            w = self._get_weight(test_weight_quantity=test_weight_quantities[idx].unsqueeze(0), cal_weight_quantities=cal_weight_quantities)
            weights.append(w)
        
        weights = torch.cat(weights)
        with open(f'temp_data/{self.similarity_type}_distances.pkl', 'wb') as handle:
            pickle.dump(weights, handle)

    def evaluate(self, cal_indices, test_indices, alpha):
        
        cal_scores, cal_labels = self.all_cal_scores[cal_indices], self.all_cal_labels[cal_indices]
        test_scores, test_labels = self.all_test_scores[test_indices], self.all_test_labels[test_indices]

        cal_scores = cal_scores[torch.arange(cal_scores.shape[0], device=cal_scores.device), cal_labels] #select only scores from labels

        cal_weight_quantities = self.all_cal_weight_quantities[cal_indices]

        test_weight_quantities = self.all_test_weight_quantities[test_indices]

        prediction_sets = []

        class_set_sizes = {
            i: [] for i in range(self.num_classes)
        }
        
        for idx, test_score in enumerate(test_scores):
            
            w = self._get_weight(test_weight_quantity=test_weight_quantities[idx].unsqueeze(0), cal_weight_quantities=cal_weight_quantities)
            
            # calculate new quantile with weights
            q_hat_reweighted = self.calculate_conformal_value(cal_scores, alpha=alpha, weights=w)

            S = self._generate_prediction_set(test_score.unsqueeze(0), q_hat_reweighted)[0]
            
            class_set_sizes[test_labels[idx].item()].append(S.sum().item())
            prediction_sets.append(S)


        prediction_sets = torch.stack(prediction_sets, dim=0)
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

    def quantile_1D(self, scores, weights, quantile):
        sort_scores, sort_idx = torch.sort(scores)
        sorted_weights = weights[sort_idx] # does nothing if weights are uniform
        weight_cdf = torch.cumsum(sorted_weights, dim=0) # weight CDF
        q_idx = torch.searchsorted(weight_cdf, quantile, right=True) - 1
        q_wcp = sort_scores[q_idx] # get the value of the quantile

        return q_wcp
    
    def calculate_conformal_value(self, scores, alpha, weights=None):
        """
        Calculate the 1-alpha quantile of scores.
        
        :param scores: non-conformity scores.
        :param alpha: a significance level.
        
        :return: the threshold which is use to construct prediction sets.
        """
        default_q_hat = torch.max(scores)
        if alpha >= 1 or alpha <= 0:
                raise ValueError("Significance level 'alpha' must be in [0,1].")
        N = scores.shape[0]
        quantile_value = math.ceil(N + 1) * (1 - alpha) / N


        if weights is None:
            weights = (torch.ones(scores.shape) / scores.shape[0]).to(scores.device)

        # return torch.quantile(scores, qunatile_value, dim=0).to(scores.device)
        return self.quantile_1D(scores=scores, weights=weights, quantile=quantile_value).to(scores.device)



class MondrianPredictorBase(BasePredictor):
    """
    Split Conformal Prediction (Vovk et al., 2005).
    Book: https://link.springer.com/book/10.1007/978-3-031-06649-8.

    Args:
        score_function: A callable or scoring object to compute nonconformity.
        model: PyTorch model.
        temperature: Temperature parameter for softmax scaling.
        num_classes: Number of classes in classification task.
        num_partitions: Number of partitions for Mondrian calibration.
        canon_cutoff: Threshold for canonicalization decisions.
        subsample: Whether to subsample calibration data.
        target_partition: Which variable to partition on ("none", "label", etc).
    """
    def __init__(
        self,
        score_function: Any,  # ideally: Protocol or base class if available
        model: Optional[nn.Module] = None,
        temperature: float = 2.0,
        num_classes: int = 10,
        num_partitions: int = 8,
        canon_cutoff: float = 0.9,
        subsample: bool = False,
        target_partition: str = "none",
        log_dir: str = ""
    ):
        super().__init__(score_function, model, temperature)

        self.num_classes = num_classes
        self.cutoff = canon_cutoff
        self.subsample = subsample
        self.target_partition = target_partition

        # map for partition to partition index
        self.partition2idx = get_partition_mapping(num_partitions)

        self.num_bins = get_num_bins(self.target_partition, self.num_classes)
        
        if log_dir == "":
            self.log_dir = f"./experiment_logs/mondrian/{target_partition}"
        else:
            self.log_dir = log_dir

        os.makedirs(self.log_dir, exist_ok=True)


    def _get_diagnostics(self, **kwargs):
        """
        Subclasses must implement this method to handle the calculation logic.
        """
        raise NotImplementedError("diagnostics not implemented.")
    

    def _collect_scores(self, dataloader):
        raise NotImplementedError("score collection not implemented.")


    def _get_partition_info(self):
        """Get target partition information for each sample. Used to calculate coverage of target partition"""
    

        if self.subsample:
            with open(self.log_dir + '/cutoff_indices.pkl', 'rb') as handle:
                indices = pickle.load(handle)
                batch_size = torch.cat(indices, dim=0).max().item() + 1
        partitions = []
        if self.target_partition != "none":
            with open(self.log_dir + f'/{self.target_partition}.pkl', 'rb') as handle:
                partition_info = pickle.load(handle)
                start_idx = 0
                i = 0
                
                partition_info = partition_info.to(indices[0].device)

                while start_idx < partition_info.shape[0]:
                    batch = partition_info[start_idx:start_idx + batch_size]
                    if self.subsample:
                        batch = batch[indices[i]]


                    partitions.append(batch)
                    start_idx += batch_size
                    i += 1
    
                    
            partitions = torch.cat(partitions).to(self._device)
            
        return partitions


    def collect_scores(self, dataloader, shifted_loader=None):
        print("Collecting scores....")
        self._model.eval()

        self.scores, self.labels, self.partition_info = self._collect_scores(dataloader)

        
        if self.target_partition != "none":
            self.target_partition_info = self._get_partition_info()

        
        print("Score collection done.")

    def calibrate(self, cal_scores_partitioned, alpha):
        q_values_partitioned = {}

        # calculate quantiles per partition
        for partition, cal_scores in cal_scores_partitioned.items():
            if cal_scores.shape[0] > 0:
                q_values_partitioned[partition] = self.calculate_conformal_value(cal_scores, alpha=alpha)
            else:
                q_values_partitioned[partition] = None
            
        
        return q_values_partitioned
        

    #############################  
    # Partitioning logic should be implemented here
    ############################
    def _partition_data(self, scores, labels, split_info):
        raise NotImplementedError("data partitioning not implemented.")


    #############################  
    # The prediction process
    ############################

    def _generate_prediction_set(self, scores, q_hat: torch.Tensor):
        """
        Generate the prediction set with the threshold q_hat.

        Args:
            scores (torch.Tensor): The non-conformity scores of {(x,y_1),..., (x,y_K)}.
            q_hat (torch.Tensor): The calibrated threshold.

        Returns:
            torch.Tensor: A tensor of 0/1 values indicating the prediction set for each example.
        """
        # if no cal samples return top label
        if q_hat is None:
            mins = scores.argmin(dim=-1)

            sets = torch.zeros(scores.shape, device=scores.device)
            sets[torch.arange(scores.shape[0], device=scores.device), mins] = 1
            return sets.int()
        else:
            return (scores <= q_hat).int()

    def predict(self, test_scores, q_hat):
        return self._generate_prediction_set(test_scores, q_hat)


    def evaluate(self, cal_indices, test_indices, alpha):
        
        cal_scores, cal_labels, cal_partition_info = self.scores[cal_indices], self.labels[cal_indices], self.partition_info[cal_indices]

        test_scores, test_labels, test_partition_info = self.scores[test_indices], self.labels[test_indices], self.partition_info[test_indices]
        
        test_target_partition_info = self.target_partition_info[test_indices]
        
        cal_scores = cal_scores[torch.arange(cal_scores.shape[0], device=cal_scores.device), cal_labels] #select only scores from labels

        cal_target_partition_info = self.target_partition_info[cal_indices]

        cal_partitions, cal_partitions_labels, _ = self._partition_data(cal_scores, cal_labels, cal_partition_info, cal_target_partition_info)



        
        # get q values per partition
        q_values_partitioned = self.calibrate(cal_partitions, alpha)

        test_partitions, test_partitions_labels, test_target_partition_info = self._partition_data(test_scores, test_labels, test_partition_info, test_target_partition_info)
        
        
        # get prediction sest per partition

        prediction_sets_partitioned = {}
        partition_set_sizes = {}


        for partition_idx, partition in test_partitions.items():
            
            # generate sets per partition with corresponding partition q_values

            if partition.shape[0] > 0:

                S = self._generate_prediction_set(partition, q_values_partitioned[partition_idx])

                partition_set_sizes[partition_idx] = [s.sum().item() for s in S]
                prediction_sets_partitioned[partition_idx] = S


        prediction_sets = []
        labels = []
        target_partition_info = []

        # aggregate the partition results
        for partition, partition_set in prediction_sets_partitioned.items():
            if len(partition_set) > 0:
                prediction_sets.append(partition_set)
                labels.append(test_partitions_labels[partition])
                target_partition_info.append(test_target_partition_info[partition])

        prediction_sets = torch.cat(prediction_sets, dim=0)
        test_labels = torch.cat(labels, dim=0)
        target_partition_info = torch.cat(target_partition_info, dim=0)

        class_set_sizes = {
            i: [] for i in range(self.num_classes)
        }

        for pred, label in zip(prediction_sets, test_labels):
            class_set_sizes[label.item()].append(pred.sum().item())    


        class_sizes = self.get_set_size_per_class(class_set_sizes)
        class_cov = self.get_coverage_per_class(test_labels, prediction_sets)
        
        partition_sizes = self.get_set_size_per_partition(partition_set_sizes=partition_set_sizes)
        partition_cov = self.get_coverage_per_partition(prediction_sets_partitioned, test_partitions_labels)

        
        res_dict = {
            "Coverage_rate": self._metric('coverage_rate')(prediction_sets, test_labels),
            "Average_size": self._metric('average_size')(prediction_sets, test_labels),
            "Class_coverage": class_cov,
            "Class_set_size": class_sizes,
            "Partition_coverage": partition_cov,
            "Partition_set_size": partition_sizes,   
        }
        
        if self.target_partition != 'none':
            
            res_dict["Target_partition_coverage"] = self.get_coverage_per_part(test_labels, prediction_sets, target_partition_info)


        return res_dict
    

    def get_coverage_per_part(self, test_labels, prediction_sets, partition_info):
        part_coverages = {
            i: [] for i in range(self.num_bins)
        }
        for i, label  in enumerate(test_labels):
            part_coverages[partition_info[i].item()].append(prediction_sets[i][label.item()].item())

        return {label: np.mean(p_coverage) for label, p_coverage in part_coverages.items()}

    def get_set_size_per_class(self, class_set_sizes):
        
        return {label: np.mean(label_set_sizes) for label, label_set_sizes in class_set_sizes.items()}

    def get_coverage_per_class(self, test_labels, prediction_sets):
        class_coverages = {
            i: [] for i in range(self.num_classes)
        }
        for i, label  in enumerate(test_labels):
            class_coverages[label.item()].append(prediction_sets[i][label.item()].item())

        return {label: np.mean(class_coverage) for label, class_coverage in class_coverages.items()}
    
    def get_set_size_per_partition(self, partition_set_sizes):
        
        return {label: np.mean(label_set_sizes) for label, label_set_sizes in partition_set_sizes.items()}

    def get_coverage_per_partition(self, prediction_sets, labels):
        partition_coverages = {
            i: [] for i in prediction_sets.keys()
        }

        for partition_name, partition_sets in prediction_sets.items():
            
            for i, part_set  in enumerate(partition_sets):

                partition_coverages[partition_name].append(part_set[labels[partition_name][i].item()].item())

        part_coverage = {part_name: np.mean(partition_coverage) for part_name, partition_coverage in partition_coverages.items()}

        return part_coverage

    def quantile_1D(self, scores, weights, quantile):
        sort_scores, sort_idx = torch.sort(scores)
        sorted_weights = weights[sort_idx] # does nothing if weights are uniform
        weight_cdf = torch.cumsum(sorted_weights, dim=0) # weight CDF
        q_idx = torch.searchsorted(weight_cdf, quantile, right=True) - 1
        # q_idx = torch.where(weight_cdf >= quant)[0][0]
        q_wcp = sort_scores[q_idx] # get the value of the quantile

        return q_wcp
    
    def calculate_conformal_value(self, scores, alpha, weights=None):
        """
        Calculate the 1-alpha quantile of scores.
        
        :param scores: non-conformity scores.
        :param alpha: a significance level.
        
        :return: the threshold which is use to construct prediction sets.
        """
        default_q_hat = torch.max(scores)
        if alpha >= 1 or alpha <= 0:
                raise ValueError("Significance level 'alpha' must be in [0,1].")
        N = scores.shape[0]
        qunatile_value = math.ceil(N + 1) * (1 - alpha) / N


        if weights is None:
            weights = (torch.ones(scores.shape) / scores.shape[0]).to(scores.device)

        # return torch.quantile(scores, qunatile_value, dim=0).to(scores.device)
        return self.quantile_1D(scores=scores, weights=weights, quantile=qunatile_value).to(scores.device)


#######################################################################################################################


class MondrianGroupPredictor(MondrianPredictorBase):
    """
    Mondrian Conformal Prediction with group-based calibration.
    Based on: Vovk et al., 2005.
    Book: https://link.springer.com/book/10.1007/978-3-031-06649-8.

    Args:
        score_function: Non-conformity scoring function.
        model: PyTorch model.
        temperature: Temperature scaling parameter.
        num_classes: Number of classification classes.
        num_partitions: Number of partitions (e.g., angles or clusters).
        canon_cutoff: Threshold for canonicalization.
        subsample: Whether to subsample calibration scores.
        target_partition: Type of partitioning ("label", "angle", etc).
        save_counts: Whether to save count stats (e.g. for analysis).
    """
    def __init__(
        self,
        score_function: Any,  # Ideally use a Protocol or scoring base class
        model: Optional[nn.Module] = None,
        temperature: float = 2.0,
        num_classes: int = 10,
        num_partitions: int = 8,
        canon_cutoff: float = 0.9,
        subsample: bool = False,
        target_partition: str = "none",
        log_dir: str = ""
    ) -> None:
        
        super().__init__(score_function, model, temperature, subsample=subsample, target_partition=target_partition, log_dir=log_dir)
       
        # self.save_counts = save_counts
        self.num_classes = num_classes
        self.cutoff = canon_cutoff

        self.num_partitions = get_num_bins(target_partition, self.num_classes)

        # map for partition to partition index
        self.partition2idx = get_partition_mapping(num_partitions)


    # GROUP
    def _get_diagnostics(self, dataloader):

        
        partition_info_list = []
        indices_list = []
        with torch.no_grad():
            for examples in dataloader:
                
                tmp_x, tmp_labels = examples[0].to(self._device), examples[1].to(self._device)
                target_partition_info = examples[2]

                g_probs = torch.softmax(self._model.canonicalizer.get_group_activations(tmp_x), dim=-1)

                probs, partition_info = torch.max(g_probs, dim=-1)

                
                indices_list.append((probs > self.cutoff).nonzero(as_tuple=False).squeeze())
                partition_info = partition_info[probs > self.cutoff]
                tmp_labels = tmp_labels[probs > self.cutoff]
                target_partition_info = target_partition_info[probs.to(target_partition_info.device) > self.cutoff]
            

                partition_info_list.append(partition_info)

        
        with open(self.log_dir + '/cutoff_indices.pkl', 'wb') as handle:
            pickle.dump(indices_list, handle)

        return partition_info_list, indices_list
    
    # GROUPs
    def _collect_scores(self, dataloader):
        logits_list = []
        labels_list = []

        # partition_info_list = [] #store any info used for partitioning
        partition_info_list, indices = self._get_diagnostics(dataloader)

        with torch.no_grad():
            for i, examples in enumerate(dataloader):

                tmp_x, tmp_labels = examples[0].to(self._device), examples[1].to(self._device)

                tmp_x, tmp_labels = tmp_x[indices[i]], tmp_labels[indices[i]]
                tmp_logits = self._logits_transformation(self._model(tmp_x)).detach()

                # partition_info = self._model.canonicalizer.get_groupelement(tmp_x)["rotation"]
                # self._model.canonicalizer.get_group_activations(tmp_x)
                # partition_info = tmp_labels

                
                logits_list.append(tmp_logits)
                labels_list.append(tmp_labels)

        logits = torch.cat(logits_list).float().to(self._device)
        labels = torch.cat(labels_list).to(self._device)

        #filter out the uncertain elements
        # logits = logits[indices]
        # labels = labels[indices]
        partition_info = torch.cat(partition_info_list).float().to(self._device) # save weight info


        scores = self.score_function(logits)

        self.num_samples = scores.shape[0]
        
        return scores, labels, partition_info


    #############################  
    # Partitioning logic should be implemented here - GROUP
    ############################
    def _partition_data(self, scores, labels, split_info, target_split_info):
        # partition the calibration data.

        partitions = {}
        partitions_labels = {}
        partitions_target_info = {}

        partition_names = list(self.partition2idx.values())

        for partition_name in partition_names:
            
            indices = torch.argwhere(split_info == partition_name)

            if indices.shape[0] > 0:
                
                partitions[partition_name] = scores[indices].squeeze(dim=1)        
                partitions_labels[partition_name] = labels[indices].squeeze(dim=1) #re-order the labels based on the partitions
                partitions_target_info[partition_name] = target_split_info[indices].squeeze(dim=1) #re-order the labels based on the partitions
            else:
                partitions[partition_name] = torch.tensor([])        
                partitions_labels[partition_name] = torch.tensor([])
                partitions_target_info[partition_name] = torch.tensor([])
        return partitions, partitions_labels, partitions_target_info

