import torch
from torch.nn import functional as F
import torch.nn as nn

class TabularMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TabularMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2*hidden_dim)
        self.fc3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        return x
    
    def get_transform(self):
        def transform(tabular_vector):
            x = torch.tensor(tabular_vector, dtype=torch.float32).squeeze(0)
            return x
        return transform