import torch

def log_barrier_penalty(state, cost, env, r_safe=0, eps=1e-6, pad = 2):
    """
    Log-barrier penalty for safety constraints.

    Supports:
      - state shape (2,)
      - state shape (N, 2)
      - state as list/tuple of (x, y)

    Args:
        state: (x,y) or (N,2)
        cost: torch tensor of shape (n_constraints,)
        env: environment
        r_safe: safety radius
    """
    state_t = torch.as_tensor(state, dtype=torch.float32)

    # Ensure shape (N,2)
    if state_t.ndim == 1:
        state_t = state_t.unsqueeze(0)
        
    N = state_t.shape[0]   # number of (x,y) tuples
    x = state_t[:, 0]
    y = state_t[:, 1]

    violated = False

    # ----- borders -----
    # right wall
    mask = x >= (env.size[0] - r_safe - pad)
    cost[0] += torch.where(mask, 10.0, 1e-4).sum()
    violated |= mask.any().item()

    # left wall
    mask = -x + r_safe - 1 + pad >= 0
    cost[1] += torch.where(mask, 10.0, 1e-4).sum()
    violated |= mask.any().item()

    # top wall
    mask = y >= (env.size[1] - r_safe - pad)
    cost[2] += torch.where(mask, 10.0, 1e-4).sum()
    violated |= mask.any().item()

    # bottom wall
    mask = -y + r_safe - 1 + pad >= 0
    cost[3] += torch.where(mask, 10.0, 1e-4).sum()
    violated |= mask.any().item()

    # ----- internal walls -----
    for i, (wx, wy) in enumerate(env.inner_walls):
        d2 = (x - wx)**2 + (y - wy)**2
        mask = d2 == eps
        cost[4 + i] += torch.where(mask, 10.0, 1e-4).sum()
        violated |= mask.any().item()

    # average over number of states
    cost = cost / N

    return cost, violated
