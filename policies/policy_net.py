import torch
import torch.nn as nn
from torch.distributions import Categorical

class PolicyNet(nn.Module):
    def __init__(self, state_size=2, action_size=4, hidden_sizes=(64, 64)):
        super(PolicyNet, self).__init__()

        # Two hidden layers with normalization and nonlinearity
        self.fc1 = nn.Linear(state_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc_out = nn.Linear(hidden_sizes[1], action_size)

        # Optional normalization for stability
        self.ln1 = nn.LayerNorm(hidden_sizes[0])
        self.ln2 = nn.LayerNorm(hidden_sizes[1])

        # Initialization
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc_out.bias)
        print("New")

    def forward(self, state):
        """
        Forward pass returning action probabilities.
        """
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        logits = self.fc_out(x)
        probs = F.softmax(logits, dim=1)
        return probs

    def act(self, state):
        """
        Same as your original version: sample from categorical distribution.
        """
        state = np.array(state, dtype=np.float32)
        state = torch.from_numpy(state).float().unsqueeze(0).to("cpu")
        probs = self.forward(state).cpu()
        model = Categorical(probs)
        action = model.sample()
        return action.item(), model.log_prob(action), model.entropy()