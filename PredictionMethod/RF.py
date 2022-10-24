#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import sklearn.ensemble as se
import numpy as np
import torch
import torch.nn as nn

class RF(nn.Module):
    type = 'RF'
    name = 'RF'
    def __init__(self, max_depth=10, num=1000, min_split=3):
        super(RF, self).__init__()
        self.model = se.RandomForestRegressor(max_depth=10, n_estimators=1000, min_samples_split=3)

    def Train(self, train_x, train_y):
        train_x = np.array(train_x)
        train_y = np.array(train_y).ravel()
        print('RF x: ', train_x.shape)
        print('RF y: ', train_y.shape)

        self.model.fit(train_x, train_y)

    def forward(self, x):
        x = np.array(x)
        result = self.model.predict(x)
        return result
