import torch

def update_constraint_value_estimates(C_tau, V_i_estimate, alpha):
    """
    Update running EMA of constraint costs (trajectory-level).

    Args:
        C_tau: tensor of shape (m,)
        V_i_estimate: list of length m (None or tensor)
        alpha: EMA rate

    Returns:
        Updated V_i_estimate
    """
    for i in range(len(C_tau)):
        Ci = C_tau[i].detach()
        if V_i_estimate[i] is None:
            V_i_estimate[i] = Ci
        else:
            V_i_estimate[i] = (1 - alpha) * V_i_estimate[i] + alpha * Ci
    return V_i_estimate


def update_constraint_value_estimates_batch(C_tau_batch, V_i_estimate, alpha):
    """
    Batch version (recommended).

    Args:
        C_tau_batch: tensor of shape (batch_size, m)
        V_i_estimate: list of length m
        alpha: EMA rate

    Returns:
        Updated V_i_estimate
    """
    batch_mean = C_tau_batch.mean(dim=0).detach()

    for i in range(len(batch_mean)):
        if V_i_estimate[i] is None:
            V_i_estimate[i] = batch_mean[i]
        else:
            V_i_estimate[i] = (1 - alpha) * V_i_estimate[i] + alpha * batch_mean[i]

    return V_i_estimate


def compute_barrier_weight(C_tau, V_i_estimate, lambda_maze):
    """
    Compute normalized barrier weight:
        sum_i lambda * (C_i / E[C_i])

    Args:
        C_tau: tensor (m,)
        V_i_estimate: list of length m
        lambda_maze: scalar

    Returns:
        scalar tensor
    """
    bw = 0.0
    for i in range(len(C_tau)):
        bw += lambda_maze * (C_tau[i] / V_i_estimate[i])
    return bw
