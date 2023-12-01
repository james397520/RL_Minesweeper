import torch
import torch.nn as nn


# DQN模型
class DQN(nn.Module):
    def __init__(self, size):
        super(DQN, self).__init__()
        self.size = size
        self.CNN = nn.Sequential(
            nn.Conv2d(15, 32, kernel_size=3, stride=1, padding=1),
            # nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.head_indx = torch.nn.Linear(self.size[0]*self.size[1], self.size[0]*self.size[1])
        self.head_flag = torch.nn.Linear(self.size[0]*self.size[1], 2)

    def forward(self, state):
        # print(state.size())
        batch_size, channels, height, width = state.size()
        state = self.CNN(state)
        state = state.view(batch_size, -1)
        idx = self.head_indx(state)
        act = self.head_flag(state)
        # act = result[:, :result.shape[1]-2, :, :]
        # flag = result[:, :-2, :, :]
        return idx, act