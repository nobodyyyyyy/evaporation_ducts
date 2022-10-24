#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import numpy as np
import torch
import torch.nn as nn

class Svr(nn.Module):
    type = 'SVR'
    name = 'SVR'
    def __init__(self, kernel='rbf', C=0.01, gamma=0.01, cv=5, grid=True):
        super(Svr, self).__init__()
        if grid is False:
            self.model = SVR(kernel=kernel, gamma=gamma, C=C)
        else:
            self.model = GridSearchCV(SVR(), param_grid={"kernel": ("linear", 'rbf', 'sigmoid'), "C": np.logspace(-3, 3, 7),
                                                "gamma": np.logspace(-3, 3, 7)}, cv=5)

    def Train(self, train_x, train_y):
        train_x = np.array(train_x)
        train_y = np.array(train_y).ravel()
        self.model.fit(train_x, train_y)

    def forward(self, x):
        x = np.array(x)
        result = self.model.predict(x)
        return result
