import torch


def initialize_constraint_vector(n_constraints, device=None):
    """
    Create zero-initialized constraint cost vector.

    Args:
        n_constraints: int
        device: torch device

    Returns:
        tensor of shape (n_constraints,)
    """
    return torch.zeros(n_constraints, dtype=torch.float32, device=device)
