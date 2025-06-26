from typing import Callable

import torch

def cross_entropy(
    predictions: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    """
    Computes the cross-entropy between target and predicted distributions.

    Args:
        predictions (torch.Tensor): Tensor of shape [..., D], representing predicted probabilities.
        target (torch.Tensor): Tensor of same shape [..., D], representing target probabilities (e.g. one-hot or soft).

    Returns:
        torch.Tensor: Tensor of shape [...] containing the cross-entropy values.
    """
    # Add small epsilon for numerical stability before taking log
    log_preds = (predictions + 1e-16).log()
    
    # Element-wise product with target, then sum over last dim (class dim)
    return -(target * log_preds).sum(dim=-1)


def total_variation_distance(
    a: torch.Tensor,
    b: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the Total Variation Distance between distribution 'a' and each distribution in 'b' using torch.norm.

    Args:
        a (torch.Tensor): A tensor of shape [1, D] representing a probability distribution.
        b (torch.Tensor): A tensor of shape [N, D] representing N probability distributions.

    Returns:
        torch.Tensor: A tensor of shape [N] containing the Total Variation Distances.
    """
    # Calculate the L1 norm of the difference between 'a' and each distribution in 'b'
    norm = torch.norm(torch.abs(a - b), p=1, dim=1) / 2

    return norm


def hellinger_distance(
    a: torch.Tensor,
    b: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the Hellinger distance between distribution 'a' and each distribution in 'b' using torch.norm.
    
    Args:
    a (torch.Tensor): A tensor of shape [1, D] representing a probability distribution.
    b (torch.Tensor): A tensor of shape [N, D] representing N probability distributions.
    
    Returns:
    torch.Tensor: A tensor of shape [N, 1] containing the Hellinger distances.
    """

    # Compute the squared Hellinger distance using torch.norm
    
    norm = torch.norm(torch.sqrt(a) - torch.sqrt(b), p=2, dim=1) / torch.sqrt(torch.tensor(2.0))
    
    return norm

def uniform(cal_weight_quantities, test_weight_quantity, w_transform=None):
    """Returns constant weight, independent of cal/test values.
    Args:
        test_weight_quantity: Tensor [1, D]: a single test distribution with support D
        cal_weight_quantity: Tensor [N_cal, D]: all calibration distributions over support D
    returns:
        Tensor [N_cal, 1]: The weights for all calibration samples for the specific test instance
    """

    weights_unnormalized = torch.ones(cal_weight_quantities.shape[0], device=cal_weight_quantities.device) #uniform weights for now
    
    return weights_unnormalized



def inverse_kl(
    cal_weight_quantities: torch.Tensor,
    test_weight_quantity: torch.Tensor,
    w_transform: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """Returns kl-divergence between test distribution and calibration distributions
    Args:
        test_weight_quantity: Tensor [1, D]: a single test distribution with support D
        cal_weight_quantity: Tensor [N_cal, D]: all calibration distributions over support D
    returns:
        Tensor [N_cal, 1]: The weights for all calibration samples for the specific test instance
    """


    kl = torch.nn.functional.kl_div((cal_weight_quantities + 1e-16).log(), (test_weight_quantity + 1e-16).log(), log_target=True, reduction="none").sum(dim=1)
    if torch.isnan(kl).any():
        raise Exception("Contains nan values")

    weights_unnormalized = 1 / w_transform(kl)

    return weights_unnormalized

def inverse_hellinger(
    cal_weight_quantities: torch.Tensor,
    test_weight_quantity: torch.Tensor,
    w_transform: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """Returns the cross entropy between test distribution and calibration distributions
    Args:
        test_weight_quantity: Tensor [1, D]: a single test distribution with support D
        cal_weight_quantity: Tensor [N_cal, D]: all calibration distributions over support D
    returns:
        Tensor [N_cal, 1]: The weights for all calibration samples for the specific test instance
    """
    hellinger = hellinger_distance(test_weight_quantity, cal_weight_quantities) #trying the reverse
    if torch.isnan(hellinger.any()):
        raise Exception("Contains nan values")
    weights_unnormalized = 1 / w_transform(hellinger)

    # weights_unnormalized = hellinger

    return weights_unnormalized

def inverse_tv(
    cal_weight_quantities: torch.Tensor,
    test_weight_quantity: torch.Tensor,
    w_transform: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """Returns the cross entropy between test distribution and calibration distributions
    Args:
        test_weight_quantity: Tensor [1, D]: a single test distribution with support D
        cal_weight_quantity: Tensor [N_cal, D]: all calibration distributions over support D
    returns:
        Tensor [N_cal, 1]: The weights for all calibration samples for the specific test instance
    """
    tvd = total_variation_distance(test_weight_quantity, cal_weight_quantities) #trying the reverse
    if torch.isnan(tvd.any()):
        raise Exception("Contains nan values")
    weights_unnormalized = 1 / w_transform(tvd)
    


    return weights_unnormalized

def inverse_crossentropy(
    cal_weight_quantities: torch.Tensor,
    test_weight_quantity: torch.Tensor,
    w_transform: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """Returns the cross entropy between test distribution and calibration distributions
    Args:
        test_weight_quantity: Tensor [1, D]: a single test distribution with support D
        cal_weight_quantity: Tensor [N_cal, D]: all calibration distributions over support D
    returns:
        Tensor [N_cal, 1]: The weights for all calibration samples for the specific test instance
    """
    cross_ent = cross_entropy(cal_weight_quantities, test_weight_quantity)
    if torch.isnan(cross_ent.any()):
        raise Exception("Contains nan values")
    weights_unnormalized = 1 / w_transform(cross_ent)
    
    # weights_unnormalized = cross_ent

    return weights_unnormalized

def inverse_l2(
    cal_weight_quantities: torch.Tensor,
    test_weight_quantity: torch.Tensor,
    w_transform: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """Returns the L2 norm between test distribution and calibration distributions
    Args:
        test_weight_quantity: Tensor [1, D]: a single test distribution with support D
        cal_weight_quantity: Tensor [N_cal, D]: all calibration distributions over support D
    returns:
        Tensor [N_cal, 1]: The weights for all calibration samples for the specific test instance
    """
    l2 = torch.norm(cal_weight_quantities - test_weight_quantity, p=2, dim=-1)
    weights_unnormalized = 1 / w_transform(l2)

    # weights_unnormalized = l2

    return weights_unnormalized 

def cyclic_distance(x, y, n):
    """
    Calculate the cyclic distance between two elements in a cyclic group of order n using PyTorch.

    Parameters:
        x (torch.Tensor): First element in the group.
        y (torch.Tensor): Second element in the group.
        n (int): Order of the cyclic group.

    Returns:
        torch.Tensor: Cyclic distance between x and y.
    """
    diff = torch.abs(x - y)
    return torch.min(diff, n - diff)
    
def discrete_distance(
    cal_weight_quantities: torch.Tensor,
    test_weight_quantity: torch.Tensor,
) -> torch.Tensor:
    """implements the logic for calculating all calibration weights for a single test sample
    Args:
        test_weight_quantity: Tensor [1, D]: a single test quantity
        cal_weight_quantity: Tensor [N_cal, D]: all calibration quantities
    returns:
        Tensor [N_cal, 1]: The weights for all calibration samples for the specific test instance
    """
    
    
    distances = cyclic_distance(cal_weight_quantities, test_weight_quantity, 4)
    
    # invert the distances
    weights_unnormalized = torch.abs(distances - 2.0)
    
    return weights_unnormalized

def binary_distance(
    cal_weight_quantities: torch.Tensor,
    test_weight_quantity: torch.Tensor,
) -> torch.Tensor:
    """implements the logic for calculating all calibration weights for a single test sample
    Args:
        test_weight_quantity: Tensor [1, D]: a single test quantity
        cal_weight_quantity: Tensor [N_cal, D]: all calibration quantities
    returns:
        Tensor [N_cal, 1]: The weights for all calibration samples for the specific test instance
    """
    
    
    weights_unnormalized = (test_weight_quantity == cal_weight_quantities)

    return weights_unnormalized
