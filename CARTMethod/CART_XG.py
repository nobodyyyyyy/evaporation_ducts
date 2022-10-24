#!/usr/bin/env python 
# -*- coding:utf-8 -*-

# A regression tree for XGBoost
# import sys
# sys.path.append('..')
import copy
import numpy as np
from CARTMethod.CART_R import caRtTree

from Ensembel.utils import readfile, Divide, TrainValidTest, Ones
from PredictionMethod.Measure import GetMAP, GetMAPE, GetRMSE, R_square

class LeastSquaresLoss():
    "Least squares loss"
    def __init__(self, y, y_hat):
        self.y = y
        self.y_hat = y_hat

    def gradient(self):
        return self.y - self.y_hat

    def hess(self):
        return np.ones_like(self.y)

class caRtXGTree(caRtTree):
    #
    def __init__(self, dataset, label, y_hat, character, layers=0, parents=None, lamda=0, gamma=0.5, max_depth=100):
        #super(caRtXGTree, self).__init__()
        #lamda is the hp of regularization, and gamma is the penalty of number of leaves
        R = np.mean(label)
        self.child = []
        self.result = 0
        self.feature = []
        self.result_name = None
        # self.divide = 0
        self.continues = None
        self.layers = layers
        self.parents = parents
        self.labels = label
        self.lamda = lamda
        self.gamma = gamma

        if parents is None:
            for i in range(len(character)):
                if character[i] is True:
                    num = 0
                    sets = {}
                    for j in range(dataset.shape[0]):
                        if j not in sets.keys():
                            sets[dataset[j][i]] = num
                            dataset[j][i] = num
                            num += 1
                        else:
                            dataset[j][i] = sets[dataset[j][i]]
                character[i] = False

        if self.layers > max_depth:
            self.result = np.mean(label - y_hat)
            return
        # 如果标签数目统一，直接成为叶子节点
        elif (label == R).all():
            self.result = np.mean(label - y_hat)
            # print(self.result)
            return
        # 否则判断信息增益率
        else:
            # 这里加上离散值
            index, value, continue_value = self.GainInfor(dataset, label, y_hat)
            if value <= gamma:
                self.result = np.mean(label - y_hat)
                return
            if character[index] is False:
                min_value = np.min(dataset[:, index].astype(np.float))
                max_value = np.max(dataset[:, index].astype(np.float))
                if min_value == continue_value or max_value == continue_value:
                    self.result = np.mean(label - y_hat)
                    # print(self.result)
                    return
        self.result = index
        self.continues = continue_value
        self.Generator(dataset, label, y_hat, index, character, self.layers, lamda, gamma, max_depth)

    def Generator(self, dataset, label, y_hat, index, character, layers, lamda, gamma, max_depth):
        k = {}
        l = {}
        hat = {}
        # 如果是离散值
        # print(character)
        # print('连续')
        k['<='] = []
        k['>'] = []
        l['<='] = []
        l['>'] = []
        hat['<='] = []
        hat['>'] = []
        for i in range(len(dataset)):
            if float(dataset[i][index]) <= float(self.continues):
                # print("<=")
                # print(dataset[i][index])
                k['<='].append(dataset[i])
                l['<='].append(label[i])
                hat['<='].append(y_hat[i])
            else:
                k['>'].append(dataset[i])
                l['>'].append(label[i])
                hat['>'].append(y_hat[i])
        for key in k:
            #print('分裂')
            #print('num of hat: ', len(hat[key]))
            self.child.append(caRtXGTree(np.array(k[key]), np.array(l[key]), np.array(hat[key]),
                                           character, layers + 1, self, lamda, gamma, max_depth))
            self.feature.append(key)
        # print(self.child)

    def Sort(self, dandl, index):
        for i in range(dandl.shape[0] - 1):
            for j in range(dandl.shape[0] - 1 - i):
                if float(dandl[j][index]) > float(dandl[j + 1][index]):
                    temp = np.copy(dandl[j])
                    dandl[j] = dandl[j + 1]
                    dandl[j + 1] = temp
        return dandl[:, :-2], dandl[:, -2], dandl[:, -1]

    def Gain(self, label, y_hat):
        loss = LeastSquaresLoss(label.reshape(-1), y_hat.reshape(-1))
        G = loss.gradient()
        H = loss.hess()
        return np.sum(G), np.sum(H)

    def GainInfor(self, DataSet, label, y_hat):
        # 按照XGboost的思想，得到一阶导G和二阶H
        G, H = self.Gain(label, y_hat)
        #print(G, H)
        diff = np.zeros(DataSet.shape[1])
        value = []
        for i in range(DataSet.shape[1]):
            #print('label shape: ', label.shape)
            #print('y_hat.shape: ', y_hat.shape)
            DataSetandLabel = np.column_stack((DataSet, label, y_hat))
            #print('DataSetandLabel: ', DataSetandLabel.shape)
            DataSet, label, y_hat = self.Sort(DataSetandLabel, i)
            Diff_continue = np.zeros(DataSet.shape[0] - 1)
            #print('lambda: ', self.lamda)
            for j in range(DataSet.shape[0] - 1):
                if DataSet[j][i] == DataSet[j+1][i]:
                    Diff_continue[j] = -float('inf')
                    continue
                else:
                    GL, HL = self.Gain(label[:j+1], y_hat[:j+1])
                    GR, HR = self.Gain(label[j+1:], y_hat[j+1:])
                    #GR = G - GL
                    #HR = H - HL
                    try:
                        Diff_continue[j] = GL ** 2/(HL + self.lamda) + GR ** 2/(HR + self.lamda) - G ** 2/(H+self.lamda) - self.gamma
                        #print(Diff_continue[j])
                    except:
                        print('HL: ', HL)
                        print('GL: ', GL)
                        print('GR: ', GR)
                        print('HR: ', HR)
            argmax_continue = np.argmax(Diff_continue)
            diff[i] = Diff_continue[argmax_continue]
            value.append(DataSet[argmax_continue][i])

        maxindex = np.argmax(diff)
        return maxindex, diff[maxindex], value[maxindex]

if __name__ == '__main__':
    filepath_duct = "..\\PredictionMethod\\data8.txt"
    data = readfile(filepath_duct, 'duct')
    x, x_min, x_max = Ones(data, 1)
    # x =data
    hist_x, fore_y = Divide(x, x, his_step=6, pre_step=1)
    train_x, train_y, valid_x, valid_y, test_x, test_y = TrainValidTest(hist_x, fore_y, valid=0.1)
    y_hat = np.zeros(len(train_y))
    character = [False for _ in range(hist_x.shape[1])]
    XG = caRtXGTree(dataset=train_x.detach().numpy(), label=train_y.detach().numpy(), y_hat=y_hat, character=character)
    test_y = test_y.detach().numpy().reshape(-1)
    test_x = test_x.detach().numpy()
    result_final = np.zeros(len(test_y))
    for i in range(len(test_y)):
        result = XG.Judge(test_x[i])
        result_final[i] = result
    result_final = result_final * (x_max - x_min) + x_min
    test_y = test_y * (x_max - x_min) + x_min
    print('RMSE: ', GetRMSE(test_y, result_final))
    print('MAPE: ', GetMAPE(test_y, result_final))
    print('MAP: ', GetMAP(test_y, result_final))
    print('R2: ', R_square(test_y, result_final))
