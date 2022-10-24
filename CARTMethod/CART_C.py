#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import DecisionTree
import numpy as np
import math
import copy
import ShowTree
class CartTree():
    # 已改进的连续值
    def __init__(self, dataset, label, theta, character, layers=0, parents=None):
        # 所有实标记属于同一类，则T为单节点树
        self.child = []
        self.result = 0
        self.feature = []
        self.result_name = None
        # self.divide = 0
        self.continues = None
        self.layers = layers
        self.parents = parents
        self.labels = label
        self.g_t = 0
        self.KMean = False
        k = {}
        # 获取当前的种类与其数目
        for i in range(len(label)):
            if label[i] not in k.keys():
                k[label[i]] = 0
            k[label[i]] += 1
        max = 0
        maxindex = 0
        # 得到占比最大的标签maxindex与数目max
        for key in k:
            if max < k[key]:
                max = k[key]
                maxindex = key
        # 数据集的数目
        if dataset.shape[1] == 1:
            self.result = maxindex
            # print(self.result)
            return
        # 如果标签数目统一，直接成为叶子节点
        elif max / len(label) == 1:
            self.result = maxindex
            # print(self.result)
            return
        # 否则判断信息增益率
        else:
            # 这里加上离散值
            index, value, continue_value = self.GiniInfor(dataset, label, character)
            if character[index] is False:
                min_value = np.min(dataset[:, index].astype(np.float))
                max_value = np.max(dataset[:, index].astype(np.float))
                if min_value == continue_value or max_value == continue_value:
                    self.result = maxindex
                    # print(self.result)
                    return
            #print('min:', min_value)
            #print('max:', max_value)

            elif value <= theta or value == 1:
                self.result = maxindex
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
            self.Generator(dataset, label, index, theta, character, self.layers)

    # 已改进的连续值
    def Generator(self, dataset, label, index, theta, character, layers):
        k = {}
        l = {}
        #如果是离散值
        #print(character)
        if character[index]:
            #print('离散')
            k['='] = []
            l['='] = []
            k['!='] = []
            l['!='] = []
            for i in range(len(dataset)):
                if dataset[i][index] == self.continues:
                    k['='].append(np.append(dataset[i][:index], dataset[i][index + 1:]))
                    l['='].append(label[i])
                else:
                    k['!='].append(dataset[i])
                    l['!='].append(label[i])
        else:
            #print('连续')
            k['<='] = []
            k['>'] = []
            l['<='] = []
            l['>'] = []
            for i in range(len(dataset)):

                if float(dataset[i][index]) <= float(self.continues):
                    print("<=")
                    print(dataset[i][index])
                    k['<='].append(dataset[i])
                    l['<='].append(label[i])
                else:
                    k['>'].append(dataset[i])
                    l['>'].append(label[i])
        for key in k:
            if key == '=':
                self.child.append(CartTree(np.array(k[key]), np.array(l[key]), theta,
                                               np.append(character[:index], character[index + 1:]), layers + 1, self))
            else:
                self.child.append(CartTree(np.array(k[key]), np.array(l[key]), theta,
                                               character, layers + 1, self))
            self.feature.append(key)
        # print(self.child)

    def Gini(self, label):
        labelscount = {}
        for i in label:
            if i not in labelscount.keys():
                labelscount[i] = 0
            labelscount[i] += 1
        gini = 1
        for key in labelscount:
            gini -= (float(labelscount[key])/len(label)) ** 2
        return gini

    # character用来表示数据是离散型的还是连续型的，True表示离散型，False表示连续性
    # 已改进连续值
    def GiniInfor(self, DataSet, label, character):
        Gini = np.zeros(DataSet.shape[1])
        #连续性与离散型的划分值
        c_d_values = []
        for i in range(DataSet.shape[1]):
            feature = {}
            labels = {}
            #如果是离散值，则分为两类，如A， B， C， 分为AB， C； AC， B；BC， A。抽取其中一个为一类，其余为另一类
            if character[i]:
                for j in range(DataSet.shape[0]):
                    if DataSet[j][i] not in feature.keys():
                        feature[DataSet[j][i]] = 0
                        labels[DataSet[j][i]] = []
                    feature[DataSet[j][i]] += 1
                    labels[DataSet[j][i]].append(label[j])
                thisGini = []
                key_list = []
                for key in feature:
                    key_list.append(key)
                    D1num = len(labels[key])
                    D2num = len(label) - D1num
                    D1label = labels[key]
                    D2label = []
                    for label_key in labels:
                        if label_key != key:
                            D2label.extend(labels[label_key])
                    G = D1num / len(label) * self.Gini(D1label) + D2num / len(label) * self.Gini(D2label)
                    thisGini.append(G)
                #print(thisGini)
                thisGini = np.array(thisGini)
                #print(thisGini)
                argmax_discrete = np.argmin(thisGini)
                #print(argmax_discrete)
                Gini[i] = thisGini[argmax_discrete]
                c_d_values.append(key_list[argmax_discrete])
                #print(c_d_values[i])
            else:
                # 先按照i的属性，从小到大排序
                DataSetandLabel = np.column_stack((DataSet, label))
                DataSet, label = self.Sort(DataSetandLabel, i)
                if DataSet[0][i] == DataSet[-1][i]:
                    Gini[i] = 1
                    c_d_values.append(float(DataSet[0][i]))
                    continue
                Gini_continue = np.zeros(DataSet.shape[0] - 1)
                for j in range(DataSet.shape[0] - 1):
                    median = float(DataSet[j][i]) / 2 + float(DataSet[j + 1][i]) / 2
                    for k in range(DataSet.shape[0]):
                        if float(DataSet[k][i]) > median:
                            break
                    front = self.Gini(label[:k])
                    back = self.Gini(label[k:])
                    Gini_continue[j] = (front * k / label.shape[0] +
                                            back * (label.shape[0] - k) / label.shape[0])
                argmin_continue = np.argmin(Gini_continue)
                Gini[i] = Gini_continue[argmin_continue]
                c_d_values.append(float(DataSet[argmin_continue][i]) / 2 + float(DataSet[argmin_continue + 1][i]) / 2)
        print(Gini)
        minindex = np.argmin(Gini)
        #print(Gini)
        #print(c_d_values)
        return minindex, Gini[minindex], c_d_values[minindex]

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
    #CART剪枝算法
    def Cartprune(self):
        Tree = [self]
        root = copy.deepcopy(self)
        NLeaf = root.GetNoLeaf()
        while len(NLeaf) != 0:
            alpha = float('inf')
            for nleaf in NLeaf:
                gini1 = self.Gini(nleaf.child[0].labels)
                gini2 = self.Gini(nleaf.child[1].labels)
                gini = self.Gini(nleaf.labels)
                _, ln = nleaf.Leafprune()
                g_t = (gini - gini1 - gini2) / (ln - 1)
                #print(nleaf.result_name)
                #print('g_t:', g_t)
                nleaf.g_t = g_t
                if alpha > g_t:
                    alpha = g_t
            NLeaf.sort(key=self.sortbylayers, reverse=False)
            for nleaf in NLeaf:
                if nleaf.g_t == alpha:
                    labelnum = {}
                    for j in range(len(nleaf.labels)):
                        if nleaf.labels[j] not in labelnum:
                            labelnum[nleaf.labels[j]] = 0
                        labelnum[nleaf.labels[j]] += 1
                    result = max(labelnum, key=labelnum.get)
                    #print('prune', nleaf.result_name)
                    nleaf.result = result
                    nleaf.child = []
                    break
            Tree.append(root)
            root = copy.deepcopy(root)
            NLeaf = root.GetNoLeaf()
        return Tree


    def Setgt(self, g_t):
        if len(self.child) == 0:
            return
        if self.parents != None:
            print(self.result_name)
            self.g_t = g_t[self.result_name]
            print(self.g_t)
        for cd in self.child:
            cd.Setgt(g_t)
    #获得g_t的值
    def Getgt(self):
        root = copy.deepcopy(self)
        g_t = {}
        leafparents = root.LeafParents()[0]
        print('parents:', leafparents)
        Lp, Ln = root.Leafprune()
        while Ln != 2:
            leafparents.child = []
            labelnum = {}
            for j in range(len(leafparents.labels)):
                if leafparents.labels[j] not in labelnum:
                    labelnum[leafparents.labels[j]] = 0
                labelnum[leafparents.labels[j]] += 1
            maxlabel = 0
            for key in labelnum:
                if labelnum[key] >= maxlabel:
                    maxlabel = labelnum[key]
                    maxkey = key
            leafparents.result = maxkey
            leafprune, leafnum = root.Leafprune()
            g_t[leafparents.result_name] = (leafprune - Lp)/(Ln-1)
            Lp = leafprune
            Ln = leafnum
            leafparents = root.LeafParents()[0]
        return g_t

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
                Gini = self.Gini(self.labels)
                #Shannon = self.ShannonEnt(self.labels)
                # print(self.labels)
                leafprune += Gini
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
                    label = self.child[0].Judge(np.append(test_datas[:self.result], test_datas[(self.result + 1):]))
                else:
                    label = self.child[1].Judge(test_datas)
            else:
                if float(test_datas[self.result]) <= self.continues:
                    label = self.child[0].Judge(test_datas)
                else:
                    label = self.child[1].Judge(test_datas)
        return label

    # 根据校验集进行后剪枝
    def Beprune(self, datasets, label):
        count = 0
        for i in range(datasets.shape[0]):
            test_label = self.Judge(datasets[i])
            if test_label == label[i]:
                count += 1
        accuracy = count / label.shape[0]
        print(accuracy)
        leafparents = self.LeafParents()
        while len(leafparents) != 0:
            thisparents = leafparents.pop(0)
            child = thisparents.child
            result = thisparents.result
            thisparents.child = []
            labelnum = {}
            for j in range(len(thisparents.labels)):
                if thisparents.labels[j] not in labelnum:
                    labelnum[thisparents.labels[j]] = 0
                labelnum[thisparents.labels[j]] += 1
            maxlabel = 0
            maxkey = 0
            for key in labelnum:
                if labelnum[key] >= maxlabel:
                    maxlabel = labelnum[key]
                    maxkey = key
            thisparents.result = maxkey
            count = 0
            for i in range(datasets.shape[0]):
                test_label = self.Judge(datasets[i])
                if test_label == label[i]:
                    count += 1
            new_ac = count / label.shape[0]
            print(new_ac)
            if new_ac >= accuracy:
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

if __name__ == '__main__':
    dataset = np.load('iris_dataset.npy')
    label = np.load('iris_label.npy')
    charaName = np.load('iris_feature.npy')
    print(charaName)
    character = DecisionTree.getdiscreteflag(dataset, np.arange(0, 4))
    train_data, test_data, _ = DecisionTree.DivideData(dataset, label)
    train_dataset = train_data[:, :-1]
    train_label = train_data[:, -1]

    test_dataset = test_data[:, :-1]
    test_label = test_data[:, -1]

    root = CartTree(dataset=train_dataset, label=train_label, character=character, theta=0.1)
    root.Nameresult(charaName)
    prc = ShowTree.plotTree(root)

    Tree = root.Cartprune()
    for tree in Tree:
        count = 0
        for i in range(len(test_label)):
            result = tree.Judge(test_dataset[i])
            if result == test_label[i]:
                count += 1
        print('NoK accuracy: ', count / len(test_label))
    #Tree = root.Cartprune()
    #for tree in Tree:
        #ShowTree.plotTree(tree)
    #print(label)
    #root.Beprune(dataset, label)
    #prc = ShowTree.plotTree(root)
#    g_t = root.CARTprune()
    #g_t = root.Getgt()
    #ShowTree.plotTree(root)
    #print(g_t)
    #root.Setgt(g_t)
    #root.PrintTree()
    #
    #ShowTree.plotTree(root)


    #root.PrintTree()

