import torch


def compute_returns(rewards, gamma):
    discounts = [gamma ** k for k in range(len(rewards))]
    return torch.tensor([
        sum(discounts[j] * rewards[j + t] for j in range(len(rewards) - t))
        for t in range(len(rewards))
    ], dtype=torch.float32)


def compute_policy_loss(log_probs, rewards, gamma, barrier_weight):
    """
    Standard REINFORCE + barrier penalty
    """
    G = compute_returns(rewards, gamma)

    loss_terms = [
        -lp * (Gt - barrier_weight)
        for lp, Gt in zip(log_probs, G)
    ]

    return torch.stack(loss_terms).sum(), G.mean()


def compute_mean_entropy(batch_data):
    """
    Mean entropy over (batch × time)
    """
    entropies = [e for traj in batch_data for e in traj["entropies"]]
    return torch.stack(entropies).mean()
