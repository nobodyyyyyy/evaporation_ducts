#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

class GRU(nn.Module):
    name = 'GRU'
    type = 'NN'
    def __init__(self, input_size=6, hidden_size=36, output_size=6, layer=1):
        super(GRU, self).__init__()
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=layer, batch_first=True)
        self.reg = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.init_weight()
    
    def init_weight(self):
        torch.nn.init.xavier_normal_(self.rnn.all_weights[0][0], gain=1)
        torch.nn.init.xavier_normal_(self.rnn.all_weights[0][1], gain=1)
        torch.nn.init.constant_(self.rnn.all_weights[0][2], 0)
        torch.nn.init.constant_(self.rnn.all_weights[0][3], 0)
        #torch.nn.init.xavier_normal_(self.reg.weight, gain=1)
        '''
        for m in self.modules():
            if isinstance(m, nn.GRU):
                torch.nn.init.xavier_normal_(m.weight, gain=1)
                print('init weight')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, 0, 1)
        '''

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
    model = GRU()
    x = torch.rand((2, 10, 6))
    y, _ = model(x, None)
    print(y.shape)
