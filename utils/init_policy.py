import torch

def initialize_policy_with_manual_probs(policy, prob_matrix):
    logits = torch.log(torch.tensor(prob_matrix, dtype=torch.float32))
    with torch.no_grad():
        policy.logits[:] = logits
