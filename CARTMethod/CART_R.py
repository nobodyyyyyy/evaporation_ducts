#!/usr/bin/env python 
# -*- coding:utf-8 -*-
#classification and regression tree

import numpy as np
import math
import copy
import numpy as np
# import sys
# sys.path.append('..')
import CARTMethod.DecisionTree
from Ensembel.utils import readfile, Divide, TrainValidTest, Ones
from PredictionMethod.Measure import GetMAP, GetMAPE, GetRMSE, R_square
#

class caRtTree():
    # 已改进的连续值
    # 回归问题需要面向全数值型数据
    def __init__(self, dataset, label, character, layers=0, parents=None, option=[1, 4], max_depth=100):
        # 所有实标记属于同一类，则T为单节点树
        # option[0] 为变化阈值，当精度变化小于option[0]时，停止生成叶子节点；
        # option[1] 为每隔节点的最小个数，当小于option[1]时，停止生成叶子节点；
        R = self.Loss(label)
        self.child = []
        self.result = 0
        self.feature = []
        self.result_name = None
        # self.divide = 0
        self.continues = None
        self.layers = layers
        self.parents = parents
        self.labels = label
        self.l_t = 0

        # "True" represents discrete, and "False" is continuously
        # 先将离散型变成连续型
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
        #大于最大深度，退出
        if self.layers > max_depth:
            self.result = np.mean(label)
            return
        # 数据集的数目
        if dataset.shape[0] <= option[1]:
            self.result = np.mean(label)
            # print(self.result)
            return
        # 如果标签数目统一，直接成为叶子节点
        elif (label == R).all():
            self.result = np.mean(label)
            # print(self.result)
            return
        # 否则判断信息增益率
        else:
            # 这里加上离散值
            index, value, continue_value = self.LossInfor(dataset, label, character)
            if character[index] is False:
                min_value = np.min(dataset[:, index].astype(np.float))
                max_value = np.max(dataset[:, index].astype(np.float))
                if min_value == continue_value or max_value == continue_value:
                    self.result = np.mean(label)
                    # print(self.result)
                    return
            #print('min:', min_value)
            #print('max:', max_value)
            elif R - value <= option[0]:
                self.result = np.mean(label)
                # print(self.result)
                return
            self.result = index

                # print('result: ', self.result)
            self.continues = continue_value
                # print('dataset:', dataset[:, index])
                # print('label:', label)
                # print('value:', value)
                # print('continue:', self.continues)
                # print('continues, ', continue_value)
                # 这里需要判断，如果是离散值，则这部分的index不需要被去掉
            self.Generator(dataset, label, index, character, self.layers, option, max_depth)

    # 已改进的连续值
    def Generator(self, dataset, label, index, character, layers, option, max_depth):
        k = {}
        l = {}
        #如果是离散值
        #print(character)
        # print('连续')
        k['<='] = []
        k['>'] = []
        l['<='] = []
        l['>'] = []
        for i in range(len(dataset)):

            if float(dataset[i][index]) <= float(self.continues):
                # print("<=")
                # print(dataset[i][index])
                k['<='].append(dataset[i])
                l['<='].append(label[i])
            else:
                k['>'].append(dataset[i])
                l['>'].append(label[i])
        for key in k:
            if key == '=':
                self.child.append(caRtTree(np.array(k[key]), np.array(l[key]),
                                               np.append(character[:index], character[index + 1:]), layers + 1, self, option, max_depth))
            else:
                self.child.append(caRtTree(np.array(k[key]), np.array(l[key]),
                                               character, layers + 1, self, option, max_depth))
            self.feature.append(key)
        # print(self.child)

    # Loss Function
    def LossInfor(self, DataSet, label, character):
        loss = np.zeros(DataSet.shape[1])
        value = []
        for i in range(DataSet.shape[1]):
            DataSetandLabel = np.column_stack((DataSet, label))
            DataSet, label = self.Sort(DataSetandLabel, i)
            Loss_continue = np.zeros(DataSet.shape[0] - 1)
            for j in range(DataSet.shape[0] - 1):
                if DataSet[j][i] == DataSet[j + 1][i]:
                    Loss_continue[j] = float('inf')
                    continue
                else:
                    R1 = self.Loss(label[:j+1])
                    R2 = self.Loss(label[j+1:])
                    Loss_continue[j] = R1 + R2
            argmin_continue = np.argmin(Loss_continue)
            loss[i] = Loss_continue[argmin_continue]
            value.append(DataSet[argmin_continue][i])

        minindex = np.argmin(loss)
        return minindex, loss[minindex], value[minindex]

    def Loss(self, label):
        mean = np.mean(label)
        R = np.sum((label - mean) ** 2)
        return R

    def Sort(self, dandl, index):
        for i in range(dandl.shape[0] - 1):
            for j in range(dandl.shape[0] - 1 - i):
                if float(dandl[j][index]) > float(dandl[j + 1][index]):
                    temp = np.copy(dandl[j])
                    dandl[j] = dandl[j + 1]
                    dandl[j + 1] = temp
        return dandl[:, :-1], dandl[:, -1]

    def PrintTree(self):
        sign = ''
        for _ in range(self.layers):
            sign += '-'
        print(sign)
        print('feature:', self.feature)
        print('classname:', self.result_name)
        print('result', self.result)
        print('layers:', self.layers)
        print('parents:', self.parents)
        print('continue', self.continues)
        # print('divide feature', self.dividefeature)
        if len(self.child) == 0:
            print('have not child')
            return
        else:
            for j in range(len(self.child)):
                self.child[j].PrintTree()
    #CART剪枝算法（Regression）
    def caRtprune(self):
        Tree = [self]
        root = copy.deepcopy(self)
        NLeaf = root.GetNoLeaf()
        while len(NLeaf) != 0:
            alpha = float('inf')
            for nleaf in NLeaf:
                loss1 = self.Loss(nleaf.child[0].labels)
                loss2 = self.Loss(nleaf.child[1].labels)
                loss = self.Loss(nleaf.labels)
                _, ln = nleaf.Leafprune()
                l_t = (loss - loss1 - loss2) / (ln - 1)
                #print(nleaf.result_name)
                #print('g_t:', g_t)
                nleaf.l_t = l_t
                if alpha > l_t:
                    alpha = l_t
            NLeaf.sort(key=self.sortbylayers, reverse=False)
            for nleaf in NLeaf:
                if nleaf.l_t == alpha:
                    result = np.mean(nleaf.result)
                    #print('prune', nleaf.result_name)
                    nleaf.result = result
                    nleaf.child = []
                    break
            Tree.append(root)
            root = copy.deepcopy(root)
            NLeaf = root.GetNoLeaf()
        return Tree

    def GetNoLeaf(self):
        Nl = []
        if len(self.child) != 0:
            if self.parents != None:
                Nl.extend([self])
            for cd in self.child:
                Nl.extend(cd.GetNoLeaf())
        else:
            return []
        #Nl = list(set(Nl))
        Nl.sort(key=self.sortbylayers, reverse=True)
        return Nl
    def LeafParents(self):
        pn = []
        if len(self.child) == 0:
            # print(self.parents)
            return [self.parents]
        else:
            for i in range(len(self.child)):
                pn.extend(self.child[i].LeafParents())
            pn = list(set(pn))
        # 按照深度排序
        pn.sort(key=self.sortbylayers, reverse=True)
        return pn

    def sortbylayers(self, object):
        return object.layers

    def Leafprune(self):
        leafprune = 0
        leafnum = 0
        if len(self.child) == 0:
            leafnum += 1
            for i in range(leafnum):
                loss = self.Loss(self.labels)
                #Shannon = self.ShannonEnt(self.labels)
                # print(self.labels)
                leafprune += loss
            return leafprune, leafnum
        else:
            for j in range(len(self.child)):
                lp, ln = self.child[j].Leafprune()
                leafprune += lp
                leafnum += ln
        return leafprune, leafnum

    def Judge(self, test_datas):
        if len(self.child) == 0:
            return self.result
        else:
            if self.feature[0] == '=':
                if test_datas[self.result] == self.continues:
                    result = self.child[0].Judge(np.append(test_datas[:self.result], test_datas[(self.result + 1):]))
                else:
                    result = self.child[1].Judge(test_datas)
            else:
                if float(test_datas[self.result]) <= self.continues:
                    result = self.child[0].Judge(test_datas)
                else:
                    result = self.child[1].Judge(test_datas)
        return result

    # 根据校验集进行后剪枝
    def Beprune(self, datasets, label):
        count = 0
        y_hat = np.zeros(len(label))
        for i in range(datasets.shape[0]):
            test_result = self.Judge(datasets[i])
            y_hat[i] = test_result
        accuracy = np.sum((test_result - y_hat) ** 2) / len(label)
        #print(accuracy)
        leafparents = self.LeafParents()
        while len(leafparents) != 0:
            thisparents = leafparents.pop(0)
            child = thisparents.child
            result = thisparents.result
            thisparents.child = []
            thisparents.result = np.mean(thisparents.labels)
            count = 0
            for i in range(datasets.shape[0]):
                test_result = self.Judge(datasets[i])
                y_hat[i] = test_result
            new_ac = np.sum((test_result - y_hat) ** 2) / len(label)
            #print(new_ac)
            if new_ac <= accuracy:
                leafparents = self.LeafParents()
                break
            else:
                thisparents.child = child
                thisparents.result = result
        return self

    def Nameresult(self, Datalabel):
        if len(self.child) == 0:
            return
        else:
            self.result_name = Datalabel[self.result]
            for i in range(len(self.feature)):
                if self.feature[i] == '=':
                    self.child[i].Nameresult(np.append(Datalabel[:self.result], Datalabel[self.result + 1:]))
                else:
                    self.child[i].Nameresult(Datalabel)
        return

    def TotalLeafandLayers(self):
        if len(self.child) == 0:
            return 1, self.layers
        else:
            leaves = 0
            layers = 0
            for i in range(len(self.child)):
                l1, l2 = self.child[i].TotalLeafandLayers()
                leaves += l1
                if layers < l2:
                    layers = l2
            return leaves, layers

    def FrontLeaves(self, l=0):
        if self.parents == None:
            return l
        else:
            for i in range(len(self.parents.child)):
                if self.parents.child[i] == self:
                    return self.parents.FrontLeaves(l)
                l1, l2 = self.parents.child[i].TotalLeafandLayers()
                l += l1

def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        dataMat.append(curLine)
    dataMat = np.array(dataMat, dtype=float)
    return dataMat

if __name__ == '__main__':
    filepath_duct = "..\\PredictionMethod\\data8.txt"
    data = readfile(filepath_duct, 'duct')
    # x, x_min, x_max = Ones(data, 1)
    x = data
    hist_x, fore_y = Divide(x, x, his_step=6, pre_step=1)
    train_x, train_y, valid_x, valid_y, test_x, test_y = TrainValidTest(hist_x, fore_y, valid=0.1)
    y_hat = np.zeros(len(train_y))
    character = [False for _ in range(hist_x.shape[1])]
    XG = caRtTree(dataset=train_x.detach().numpy(), label=train_y.detach().numpy(), character=character)
    test_y = test_y.detach().numpy().reshape(-1)
    test_x = test_x.detach().numpy()
    result_final = np.zeros(len(test_y))
    for i in range(len(test_y)):
        result = XG.Judge(test_x[i])
        result_final[i] = result
    # result_final = result_final * (x_max - x_min) + x_min
    # test_y = test_y * (x_max - x_min) + x_min
    print('RMSE: ', GetRMSE(test_y, result_final))
    print('MAPE: ', GetMAPE(test_y, result_final))
    print('MAP: ', GetMAP(test_y, result_final))
    print('R2: ', R_square(test_y, result_final))
    #ShowTree.plotTree(caRt)
