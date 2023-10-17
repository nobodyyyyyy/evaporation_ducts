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


# class LSTM(nn.Module):
#
#     # https://cloud.tencent.com/developer/article/2148347
#
#     def __init__(self, input_size=1, hidden_size=8, num_layers=2, output_size=1, batch_size=4):
#         super().__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.output_size = output_size
#         self.num_directions = 1  # 单向LSTM
#         self.batch_size = batch_size
#         self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
#         self.linear = nn.Linear(self.hidden_size, self.output_size)
#
#     def forward(self, input_seq):
#         # h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size)
#         # c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size)
#         seq_len = input_seq.shape[1]  # (5, 30)
#         # input(batch_size, seq_len, input_size)
#         # input_seq = input_seq.view(self.batch_size, seq_len, 1)  # (5, 30, 1)  应该不需要这一步
#         # output(batch_size, seq_len, num_directions * hidden_size)
#         output, _ = self.lstm(input_seq)  # (8, 8)
#         # output = output.contiguous().view(self.batch_size * seq_len, self.hidden_size)  # (5 * 30, 64)
#         pred = self.linear(output)
#         # pred = pred.view(self.batch_size, seq_len, -1)
#         # pred = pred[:, -1]
#         return pred


class RNN(nn.Module):
    # https://blog.csdn.net/weixin_41555408/article/details/107150046
    def __init__(self, input_size=1, hidden_size=8, num_layers=2, output_size=1, batch_size=4):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1  # 单向LSTM
        self.batch_size = batch_size
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,  # rnn hidden unit
            num_layers=num_layers,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        # x包含很多时间步，比如10个时间步的数据可能一起放进来了，但h_state是最后一个时间步的h_state，r_out包含每一个时间步的output
        time_step = x.shape[1]
        r_out, h_state = self.rnn(x, h_state)
        #  r_out.shape: torch.Size([50, 10, 32])
        #  h_state.shape: torch.Size([1, 50, 32])
        # outs = []  # save all predictions
        # for ts in range(time_step):
        #     #         for time_step in range(r_out.size(1)):    # calculate output for each time step
        #     outs.append(self.out(r_out[:, ts, :]))
        out = self.out(r_out)[:, 0, :].unsqueeze(1)
        # out = torch.stack(outs, dim=1)
        # print(" outs: {}".format((torch.stack(outs, dim=1)).shape))  # outs: torch.Size([50, 10, 1])
        return out, h_state


class LSTMnetwork(nn.Module):

    # https://www.kaggle.com/code/ranxi169/rnn-in-pytorch?
    def __init__(self, input_size=1, hidden_size=100, output_size=1, specify=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.hidden = (torch.zeros(1, 1, self.hidden_size),
                       torch.zeros(1, 1, self.hidden_size))

    def forward(self, seq):
        out, self.hidden = self.lstm(seq.view(len(seq), 1, -1), self.hidden)
        pred = self.linear(out.view(len(seq), -1))
        return pred[-1]


class GRU(nn.Module):
    def __init__(self, feature_size=1, hidden_size=100, num_layers=1, output_size=1):
        super(GRU, self).__init__()
        self.feature_size = feature_size
        self.output_size = output_size
        self.hidden_size = hidden_size  # 隐层大小
        self.num_layers = num_layers  # gru层数
        # feature_size为特征维度，就是每个时间点对应的特征数量，这里为1
        self.gru = nn.GRU(feature_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        batch_size = x.shape[0]  # 获取批次大小

        # 初始化隐层状态
        # if hidden is None:
        #     # h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        #     # h_0 = torch.empty(self.num_layers, self.feature_size, self.output_size, dtype=torch.float)
        #     h_0 = torch.empty(self.num_layers, batch_size, self.hidden_size, dtype=torch.float)
        # else:
        #     h_0 = hidden

        # out, h_0 = self.gru(x.view(len(x), 1, -1), h_0)
        out, _ = self.gru(x.view(len(x), 1, -1))
        pred = self.fc(out.view(len(x), -1))
        return pred[-1]

        # # GRU运算
        # output, h_0 = self.gru(x, h_0)
        # # 获取GRU输出的维度信息
        # batch_size, timestep, hidden_size = output.shape
        # # 将output变成 batch_size * timestep, hidden_dim
        # output = output.reshape(-1, hidden_size)
        # # 全连接层
        # output = self.fc(output)  # 形状为batch_size * timestep, 1
        # # 转换维度，用于输出
        # output = output.reshape(timestep, batch_size, -1)
        # # 我们只需要返回最后一个时间片的数据即可
        # return output[-1]
