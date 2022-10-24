#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np

class LSTM(nn.Module):
    name = 'LSTM'
    type = 'NN'
    def __init__(self, input_size=6, hidden_size=12, output_size=6, num_layers=1):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.reg = nn.Linear(hidden_size, output_size)
        self.init_weight()

    def init_weight(self):
        torch.nn.init.xavier_normal_(self.rnn.all_weights[0][0], gain=1)
        torch.nn.init.xavier_normal_(self.rnn.all_weights[0][1], gain=1)
        torch.nn.init.constant_(self.rnn.all_weights[0][2], 0)
        torch.nn.init.constant_(self.rnn.all_weights[0][3], 0)

    def forward(self, x, state):
        x, state = self.rnn(x, state)
        x = x.contiguous()
        #print(x.shape)
        b, s, h = x.shape
        x = x.view(s*b, h)
        x = self.reg(x)
        x = x.view(b, s, -1)
        x = x[:, -1, :].unsqueeze(1)
        #print(x.shape)
        return x, state

if __name__ == "__main__":
    num_input = 6
    model = LSTM()
    x = torch.rand((2, 10, 6))
    y,_ = model(x, None)
    print(y.shape)

