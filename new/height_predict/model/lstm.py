import torch
from torch import nn


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        out = out[:, -1, :]
        out = torch.unsqueeze(out, dim=1)
        return out

    def forward_with_c(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out, _