import torch
import torch.nn as nn
from torch.distributions import Categorical

class SoftmaxPolicy(nn.Module):
    def __init__(self, width, height, num_actions=4):
        super().__init__()
        self.width  = width
        self.height = height
        self.num_actions = num_actions
        
        # A logit vector per state (H × W × A)
        # This is the analogue of policy.W in the paper
        self.logits = nn.Parameter(torch.zeros(height, width, num_actions))

    def forward(self, state):
        """
        state: tensor of shape (2,) or (B,2) with (y,x) positions
        returns: probs of shape (num_actions,) or (B,num_actions)
        """
        state_t = state
        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)  # -> (1,2)

        ys = state_t[:, 0].long()
        xs = state_t[:, 1].long()

        # (B, A)
        logits = self.logits[ys, xs]
        probs = torch.softmax(logits, dim=-1)
        return probs

    def act(self, state):
        """
        state = (y, x) grid position
        returns (action, log_prob, entropy)
        """
        y, x = state
        if y < 0 or y >= self.height or x < 0 or x >= self.width:
            # return a random safe action
            dummy_probs = torch.ones(self.num_actions) / self.num_actions
            dist = Categorical(dummy_probs)
            a = dist.sample()
            return a.item(), dist.log_prob(a), dist.entropy()
        else :
            logits = self.logits[y, x]                    # shape [num_actions]
            probs  = torch.softmax(logits, dim=0)
            dist   = Categorical(probs)
            a      = dist.sample()
            return a.item(), dist.log_prob(a), dist.entropy()

    def greedy_act(self, state):
        """Returns the argmax action without sampling."""
        y, x = state
        logits = self.logits[y, x]
        return torch.argmax(logits).item()

    def parameters(self):
        return [self.logits]