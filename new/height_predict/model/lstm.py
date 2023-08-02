import torch
from torch import nn


# class LSTM(nn.Module):
#
#     def __init__(self, input_size, hidden_size, num_layers, output_size=1):
#         super(LSTM, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False)
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         out, _ = self.lstm(x)
#         out = self.fc(out)
#         out = out[:, -1, :]
#         out = torch.unsqueeze(out, dim=1)
#         return out
#
#     def forward_with_c(self, x):
#         out, _ = self.lstm(x)
#         out = self.fc(out)
#         return out, _


# class LSTM(nn.Module):
#
#     def __init__(self, input_size=1, hidden_size=8, num_layers=2, output_size=1):
#         super().__init__()
#         self.num_layers = num_layers
#         self.hidden_size = hidden_size
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
#         self.linear = nn.Linear(hidden_size, output_size)
#         # (seq, batch, feature)
#         # (batch, seq, feature) if batch_first = True
#
#     def forward(self, x):
#         x = x.unsqueeze(0)
#         # h0 和 c0也可以不指定，默认值即全为0
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
#         out, _ = self.lstm(x, (h0, c0))
#         out = self.linear(out[0])
#         return out


class LSTM(nn.Module):

    # https://cloud.tencent.com/developer/article/2148347

    def __init__(self, input_size=1, hidden_size=8, num_layers=2, output_size=1, batch_size=4):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1  # 单向LSTM
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        # h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size)
        # c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size)
        seq_len = input_seq.shape[1]  # (5, 30)
        # input(batch_size, seq_len, input_size)
        # input_seq = input_seq.view(self.batch_size, seq_len, 1)  # (5, 30, 1)  应该不需要这一步
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq)  # (8, 8)
        # output = output.contiguous().view(self.batch_size * seq_len, self.hidden_size)  # (5 * 30, 64)
        pred = self.linear(output)
        # pred = pred.view(self.batch_size, seq_len, -1)
        # pred = pred[:, -1]
        return pred

