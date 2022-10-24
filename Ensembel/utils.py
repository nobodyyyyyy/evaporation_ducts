#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
import torch
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import pacf


def readfile(filepath, type):
    data = []
    file = open(filepath, 'r')
    for line in file:
        line_data = line.split(',')
        if type == 'x':
            line_data = line_data[146:-3]
        elif type == 'duct':
            data = line_data
        else:
            data.append(line_data)
    data = np.array(data, dtype=np.float)
    file.close()
    return data

def TrainValidTest(x, y, train=0.8, valid=None):
    train_len = int(x.shape[0] * train)
    if valid is None:
        train_x = torch.tensor(x[:train_len], dtype=torch.float32)
        train_y = torch.tensor(y[:train_len], dtype=torch.float32)
        test_x = torch.tensor(x[train_len:], dtype=torch.float32)
        test_y = torch.tensor(y[train_len:], dtype=torch.float32)
        return train_x, train_y, test_x, test_y
    else:
        valid_len = int(x.shape[0] * valid)
        train_x = torch.tensor(x[:train_len], dtype=torch.float32)
        train_y = torch.tensor(y[:train_len], dtype=torch.float32)
        valid_x = torch.tensor(x[train_len:train_len + valid_len], dtype=torch.float32)
        valid_y = torch.tensor(y[train_len:train_len + valid_len], dtype=torch.float32)
        test_x = torch.tensor(x[train_len + valid_len:], dtype=torch.float32)
        test_y = torch.tensor(y[train_len + valid_len:], dtype=torch.float32)
        return train_x, train_y, valid_x, valid_y, test_x, test_y

def Divide(x, y, his_step, pre_step):
    x_shape = len(x.shape)
    if x_shape == 3:
        hist_x = np.zeros((x.shape[0] - his_step - pre_step + 1, x.shape[1], his_step, x.shape[2]))
        fore_y = np.zeros((y.shape[0] - his_step - pre_step + 1, y.shape[1], pre_step))
    elif x_shape == 2:
        hist_x = np.zeros((x.shape[0] - his_step - pre_step + 1, x.shape[1], his_step))
        fore_y = np.zeros((y.shape[0] - his_step - pre_step + 1, y.shape[1], pre_step))
    elif x_shape == 1:
        hist_x = np.zeros((x.shape[0] - his_step - pre_step + 1, his_step))
        fore_y = np.zeros((y.shape[0] - his_step - pre_step + 1, pre_step))
    for i in range(x.shape[0] - his_step - pre_step + 1):
        for j in range(his_step):
            if x_shape == 3:
                hist_x[i, :, j, :] = x[i + j, :, :]
            elif x_shape == 2:
                hist_x[i, :, j] = x[i + j, :]
            elif x_shape == 1:
                hist_x[i, j] = x[i + j]
        for j in range(pre_step):
            if x_shape == 1:
                fore_y[i, j] = y[i + his_step + j]
            else:
                fore_y[i, :, j] = y[i + his_step + j, :]
    print(hist_x.shape)
    print(fore_y.shape)
    return hist_x, fore_y

def Ones(DataSet, dim=2):

    DataSet_min = np.min(DataSet, axis=0)
    DataSet_max = np.max(DataSet, axis=0)
    #print(DataSet_min)
    #print(DataSet_max)
    if dim == 2:
        for i in range(DataSet.shape[1]):
            DataSet[:, i] = (DataSet[:, i] - DataSet_min[i]) / (DataSet_max[i] - DataSet_min[i])
    else:
        DataSet = (DataSet - DataSet_min) / (DataSet_max - DataSet_min)
    #DataSet = (DataSet - DataSet_min) / (DataSet_max - DataSet_min)
    #DataSet = torch.tensor(DataSet)
    return DataSet, DataSet_min, DataSet_max

#自相关系数的计算
def Cal_acf(x, K=[1, 12]):
    k = K[0]
    acf = []
    mean = np.mean(x)
    #分母
    De = np.sum((x - mean) ** 2)
    while k <= K[1]:
        num = len(x) - k
        acf_k = 0
        for n in range(num):
            acf_k += (x[n] - mean) * (x[n+k] - mean)
        acf.append(acf_k)
        k += 1
    acf = np.array(acf) / De
    return acf

def Cal_dacf(x, K=[1, 12]):
    dacf = []
    k = K[0]
    while k <= K[1]:
        acf = Cal_acf(x, K=[K[0], k])
        D = np.eye(k)
        acf_d = acf[:-1]
        for i in range(k):
            D[i, :i] = acf_d[:i][::-1]
            D[i, i + 1:] = acf_d[:k - i - 1]
        D_k = np.eye(k)
        for i in range(k):
            D_k[i, :i] = acf_d[:i][::-1]
            if i != k - 1:
                D_k[i, i + 1:] = np.repeat(acf_d[i], k - i - 1)
        D_k[k - 1, k - 1] = acf[-1]
        dacf.append(np.linalg.det(D_k)/np.linalg.det(D))
        k +=1
    print(dacf)
    return np.array(dacf)

def diff(x, d=1):
    for i in range(d):
        f_x = x[:-1]
        b_x = x[1:]
        x = b_x - f_x
    return x

def ShowPrediction(source, title=''):
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure()
    plt.rcParams.update({"font.size": 20})
    plt.title(title, fontsize=24)
    x = np.arange(1, len(source) + 1)
    plt.plot(x, np.array(source), color='b', linewidth=2, markersize=7,
             label='Duct')
    plt.xlabel("times", verticalalignment="top", fontsize=20)
    plt.ylabel("value", horizontalalignment="left", fontsize=20)
    plt.legend()
    plt.show()

def Stationary_A(x, dif=3):
    for i in range(dif):
        result = adfuller(x)
        print(result)
        test_result = result[0]
        p_value = result[1]
        reject_1 = result[4]['1%']
        reject_5 = result[4]['5%']
        reject_10 = result[4]['10%']
        if p_value < 0.05 and test_result < reject_1 and test_result < reject_5 and test_result < reject_10:
            return True, i
        else:
            x = diff(x)
    return False, dif

if __name__ == "__main__":
    filepath_x = "F:\\桌面缓存文件\\蒸发波导代码\\trainx.txt"
    filepath_y = "F:\\桌面缓存文件\\蒸发波导代码\\trainy.txt"
    filepath_duct = "F:\\桌面缓存文件\\蒸发波导代码\\data8.txt"
    data = readfile(filepath_duct, 'duct')
    print(Stationary_A(data))
    print(pacf(data, 12))
    #Cal_dacf(data)


