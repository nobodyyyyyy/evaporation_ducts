#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import numpy as np

def Ones(DataSet):
    DataSet_min = min(DataSet)
    DataSet_max = max(DataSet)
    DataSet = (DataSet - DataSet_min) / (DataSet_max - DataSet_min)
    return DataSet, DataSet_min, DataSet_max

def GetRMSE(y, y_hat):
    y = y.ravel()
    y_hat = y_hat.ravel()
    RMSE = (y - y_hat) ** 2
    RMSE = sum(RMSE)/len(RMSE)
    RMSE = RMSE ** (1/2)
    #print('fun: GetRMSE')
    RMSE = np.array(RMSE)
    return RMSE

def GetMAPE(y, y_hat):
    y = y.ravel()
    y_hat = y_hat.ravel()
    index = []
    for i in range(len(y)):
        if y[i] == 0:
            index.append(i)
    y = np.delete(y, index)
    y_hat = np.delete(y_hat, index)
    MAPE = np.abs((y - y_hat)/y)
    MAPE = sum(MAPE)/len(MAPE)
    #print('fun: GetMAPE')
    MAPE = np.array(MAPE)
    return MAPE

def GetMAP(y, y_hat):
    y = y.ravel()
    y_hat = y_hat.ravel()
    MAP = np.abs((y - y_hat))
    MAP = sum(MAP)/len(MAP)
    MAP = np.array(MAP)
    return MAP

def R_square(y, y_hat):
    y_hat = y_hat.ravel()
    y = y.ravel()
    mean = np.mean(y)
    r = 1 - np.sum((y - y_hat) ** 2)/np.sum((y-mean) ** 2)
    return r

def TrainValidTest(x, y, train=0.85, valid=None):
    train_len = int(x.shape[0] * train)
    if valid is None:
        train_x = x[:train_len].clone().detach()
        train_y = y[:train_len].clone().detach()
        test_x = x[train_len:].clone().detach()
        test_y = y[train_len:].clone().detach()
        return train_x, train_y, test_x, test_y
    else:
        valid_len = int(x.shape[0] * valid)
        train_x = torch.tensor(x[:train_len], dtype=torch.float32)
        train_y = torch.tensor(y[:train_len], dtype=torch.float32)
        valid_x = torch.tensor(x[train_len:train_len+valid_len], dtype=torch.float32)
        valid_y = torch.tensor(y[train_len:train_len+valid_len], dtype=torch.float32)
        test_x = torch.tensor(x[train_len+valid_len:], dtype=torch.float32)
        test_y = torch.tensor(y[train_len+valid_len:], dtype=torch.float32)
        return train_x, train_y, valid_x, valid_y, test_x, test_y

def preprocess_data(x, y, feature_num, seq_len, pre_len):
    global y_min
    global y_max
    y_ones, y_min, y_max = DivideTest.Ones(np.array(torch.tensor(y).view(-1), dtype=float))
    source_y = torch.tensor(y_ones).unsqueeze(1)
    hist_x, fore_y = DivideTest.Divide(source_y, source_y, seq_len, pre_len)
    return TrainValidTest(hist_x, fore_y)
    #train_x, train_y, test_x, test_y = Main.TrainValidTest(hist_x, fore_y)
