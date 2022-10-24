#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
import math
import CARTMethod.ShowTree
import random
import os
def createDataSet(dataSet = None):
    if dataSet is None:
        dataSet = [[0, 0, 0, 0, 'no'],                        #数据集
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    dataSet = np.array(dataSet)
    data = []
    label = []
    for i in range(dataSet.shape[0]):
        data.append(dataSet[i][1:dataSet.shape[1]-1])
        label.append(dataSet[i][-1])
    data = np.array(data)
    label = np.array(label)
    return data, label
def getDataSetinFile(filepath):
    files = open(filepath, encoding='UTF-8')
    lines = files.readlines()
    dataset = []
    for line in lines:
        d = line.rstrip()
        d = d.split(',')
        #d[7] = float(d[7])
        #d[8] = float(d[8])
        dataset.append(d)
    return dataset
#Ture表示离散，False表示连续
def getdiscreteflag(dataset, f=[]):
    flag = []
    if len(f) == 0:
        for _ in range(dataset.shape[1]):
            flag.append(True)
        return flag
    else:
        for i in range(dataset.shape[1]):
            if i in f:
                flag.append(False)
            else:
                flag.append(True)
        return flag
class DicisionTree():
    #已改进的连续值
    #Kmean 表示 连续值的处理有KMean方法获得聚类
    def __init__(self, dataset, label, flag, theta, character, layers=0, parents=None, KMean=False):
        # 所有实标记属于同一类，则T为单节点树
        self.child = []
        self.result = 0
        self.feature = []
        self.result_name = None
        #self.divide = 0
        self.continues = None
        self.layers = layers
        self.parents = parents
        self.labels = label
        #是否使用KMean的连续值分类方法
        self.KMean = KMean
        k = {}
        #获取当前的种类与其数目
        for i in range(len(label)):
            if label[i] not in k.keys():
                k[label[i]] = 0
            k[label[i]] += 1
        max = 0
        maxindex = 0
        #得到占比最大的标签maxindex与数目max
        for key in k:
            if max < k[key]:
                max = k[key]
                maxindex = key
        #数据集的数目
        if dataset.shape[1] == 1:
            self.result = maxindex
            print(1)
            #print(self.result)
            return
        #如果标签数目统一，直接成为叶子节点
        elif max/len(label) == 1:
            self.result = maxindex
            print(2)
            #print(self.result)
            return
        #否则判断信息增益率
        else:
            #这里加上离散值
            index, value, continue_value = self.MutualInfor(dataset, label, character, flag, KMean)
            print('index', index)
            print('shannon', value)
            print('continues, ', continue_value)
            #print('C4.5:', value)W
            if KMean is False and continue_value is not None:
                print(3)
                print(dataset[:, index])
                min_value = np.min(dataset[:, index].astype(np.float))
                max_value = np.max(dataset[:, index].astype(np.float))

                if continue_value == min_value or continue_value == max_value:
                    self.result = maxindex
                    return
                print(3.1)
            elif value < theta:
                self.result = maxindex
                print(4)
                #print(self.result)
                return
            print(5)
            self.result = index
            self.continues = continue_value
            print('continues, ', continue_value)
            # 这里需要判断，如果是离散值，则这部分的index不需要被去掉
            self.Generator(dataset, label, index, flag, theta, character, self.layers, KMean)

    #已改进的连续值
    def Generator(self, dataset, label, index, flag, theta, character, layers, KMean=False):
        k = {}
        l = {}
        if self.continues is None:
            for i in range(len(dataset)):
                if dataset[i][index] not in k:
                    k[dataset[i][index]] = []
                    l[dataset[i][index]] = []
                k[dataset[i][index]].append(np.append(dataset[i][:index], dataset[i][index + 1:]))
                l[dataset[i][index]].append(label[i])
        elif KMean:
            for i in range(len(self.continues)):
                k[self.continues[i]] = []
                l[self.continues[i]] = []
            for i in range(len(dataset)):
                argmin = np.argmin(np.abs(np.array(self.continues) - float(dataset[i][index])))
                k[self.continues[argmin]].append(np.append(dataset[i][:index], dataset[i][index + 1:]))
                l[self.continues[argmin]].append(label[i])
        else:
            k['<='] = []
            k['>'] = []
            l['<='] = []
            l['>'] = []
            for i in range(len(dataset)):
                if float(dataset[i][index]) <= self.continues:
                    k['<='].append(dataset[i])
                    l['<='].append(label[i])
                else:
                    k['>'].append(dataset[i])
                    l['>'].append(label[i])
        for key in k:
            if self.continues is None:
                self.child.append(DicisionTree(np.array(k[key]), np.array(l[key]), flag, theta,
                                           np.append(character[:index], character[index+1:]), layers + 1, self, KMean))
            elif KMean:
                self.child.append(DicisionTree(np.array(k[key]), np.array(l[key]), flag, theta,
                                           np.append(character[:index], character[index+1:]), layers + 1, self, KMean))
            else:
                self.child.append(DicisionTree(np.array(k[key]), np.array(l[key]), flag, theta,
                                               character, layers + 1, self, KMean))
            self.feature.append(key)
        #print(self.child)
    def ShannonEnt(self, label):
        num = len(label)
        labelscount = {}
        for i in label:
            if i not in labelscount.keys():
                labelscount[i] = 0
            labelscount[i] += 1
        shannonEnt = 0.0
        for key in labelscount:
            prob = float(labelscount[key]) / num
            shannonEnt -= prob * math.log(prob, 2)
        return shannonEnt

    # flag 用来表示是选用信息增益筛选特征，还是信息增益比
    # True 表示信息增益， False 表示信息增益比， 默认为True,
    # character用来表示数据是离散型的还是连续型的，True表示离散型，False表示连续性
    # 已改进连续值
    #这里应该还有控制Kmeans中心点阈值，停止聚类的超参，我在这里先设为0.01
    def MutualInfor(self, DataSet, label, character, flag=True, KMean=False):
        shannon = self.ShannonEnt(label)
        benefit = np.zeros(DataSet.shape[1])
        benefit += shannon
        continue_values = np.zeros(DataSet.shape[1])
        continue_cluster = []
        for i in range(DataSet.shape[1]):
            feature = {}
            labels = {}
            if character[i]:
                for j in range(DataSet.shape[0]):
                    if DataSet[j][i] not in feature.keys():
                        feature[DataSet[j][i]] = 0
                        labels[DataSet[j][i]] = []
                    feature[DataSet[j][i]] += 1
                    labels[DataSet[j][i]].append(label[j])
                for key in feature:
                    prob = float(feature[key]) / DataSet.shape[0]
                    benefit[i] -= prob * self.ShannonEnt(labels[key])
                continue_cluster.append(None)
            elif KMean:
                center_point, bef = self.K_means_continue(DataSet, label, i, 0.01)
                benefit[i] = bef
                continue_cluster.append(center_point)
            else:
                #先按照i的属性，从小到大排序
                DataSetandLabel = np.column_stack((DataSet, label))
                DataSet, label = self.Sort(DataSetandLabel, i)
                if DataSet[0][i] == DataSet[-1][i]:
                    benefit[i] = 0
                    continue_values[i] = DataSet[0][i]
                    continue
                benefit_continue = np.zeros(DataSet.shape[0]-1)
                benefit_continue += shannon
                for j in range(DataSet.shape[0]-1):
                    median = float(DataSet[j][i])/2 + float(DataSet[j+1][i])/2
                    for k in range(DataSet.shape[0]):
                        if float(DataSet[k][i]) > median:
                            break
                    front = self.ShannonEnt(label[:k])
                    back = self.ShannonEnt(label[k:])
                    benefit_continue[j] -= (front * k/label.shape[0] +
                                            back * (label.shape[0]-k)/label.shape[0])
                argmax_continue = np.argmax(benefit_continue)
                benefit[i] = benefit_continue[argmax_continue]
                continue_values[i] = float(DataSet[argmax_continue][i])/2 + float(DataSet[argmax_continue+1][i])/2
        if flag is False:
            if shannon == 0:
                benefit = benefit * 0
            else:
                benefit = benefit / shannon
        maxindex = np.argmax(benefit)
        if character[maxindex]:
            return maxindex, benefit[maxindex], None
        #print(maxindex, ": ", benefit[maxindex])
        elif KMean:
            return maxindex, benefit[maxindex], continue_cluster[maxindex]
        else:
            return maxindex, benefit[maxindex], continue_values[maxindex]
    def Sort(self, dandl, index):
        for i in range(dandl.shape[0]-1):
            for j in range(dandl.shape[0]-1-i):
                if float(dandl[j][index]) > float(dandl[j+1][index]):
                    temp = np.copy(dandl[j])
                    dandl[j] = dandl[j+1]
                    dandl[j+1] = temp
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
        #print('divide feature', self.dividefeature)
        if len(self.child) == 0:
            print('have not child')
            return
        else:
            for j in range(len(self.child)):
                self.child[j].PrintTree()
    #根据复杂度进行预剪枝
    def Preprune(self, alpha):
        leafparents = self.LeafParents()
        print('parents:', leafparents)
        while len(leafparents) != 0:
            print(len(leafparents))
            leafprune, leafnum = self.Leafprune()
            CT_alpha = leafprune + alpha * leafnum
            print(CT_alpha)
            while len(leafparents) != 0:
                thisparents = leafparents.pop(0)
                child = thisparents.child
                thisparents.child = []
                leafprune, leafnum = self.Leafprune()
                CT_alpha_new = leafprune + alpha * leafnum
                print(CT_alpha_new)
                if CT_alpha_new <= CT_alpha:
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
                    leafparents = self.LeafParents()
                    break
                else:
                    thisparents.child = child
    def LeafParents(self):
        pn = []
        if len(self.child) == 0:
            #print(self.parents)
            return [self.parents]
        else:
            for i in range(len(self.child)):
                pn.extend(self.child[i].LeafParents())
            pn = list(set(pn))
        #按照深度排序
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
                num = self.labels.shape[0]
                Shannon = self.ShannonEnt(self.labels)
                #print(self.labels)
                leafprune += num * Shannon
            return leafprune, leafnum
        else:
            for j in range(len(self.child)):
                lp, ln = self.child[j].Leafprune()
                leafprune += lp
                leafnum += ln
        return leafprune, leafnum
    def Judge(self, test_datas):
        label = 0
        if len(self.child) == 0:
            return self.result
        else:
            if self.continues is None:
                for i in range(len(self.feature)):
                    if test_datas[self.result] == self.feature[i]:
                        label = self.child[i].Judge(np.append(test_datas[:self.result], test_datas[(self.result+1):]))
            elif self.KMean:
                #print('test_data: ', test_datas[self.result])
                #print(np.array(self.continues))
                argmin = np.abs(float(test_datas[self.result]) - np.array(self.continues, dtype='float64'))
                argmin = np.argmin(argmin)
                label = self.child[argmin].Judge(np.append(test_datas[:self.result], test_datas[(self.result+1):]))
            else:
                if float(test_datas[self.result]) <= self.continues:
                    label = self.child[0].Judge(test_datas)
                else:
                    label = self.child[1].Judge(test_datas)
        return label
    #根据校验集进行后剪枝
    def Beprune(self, datasets, label):
        count = 0
        for i in range(datasets.shape[0]):
            test_label = self.Judge(datasets[i])
            if test_label == label[i]:
                count += 1
        accuracy = count/label.shape[0]
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
            new_ac = count/label.shape[0]
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
            if self.continues is None:
                for i in range(len(self.child)):
                    self.child[i].Nameresult(np.append(Datalabel[:self.result], Datalabel[self.result+1:]))
            else:
                for i in range(len(self.child)):
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
    #统计list中最多的元素
    def maxinlist(self, list):
        l = {}
        for i in list:
            if i not in l.keys():
                l[i] = 0
            l[i] += 1

        maxkey = 0
        maxnum = 0
        for key in l:
            if maxnum == l[key] and maxnum != 0:
                return None
            elif maxnum < l[key]:
                maxkey = key
                maxnum = l[key]
        return maxkey
    def K_means(self, dataset, label, index, theta, k=2):
        shannon = self.ShannonEnt(label)
        k_shannons = []
        datasetandLabel = np.column_stack((dataset, label))
        dataset, label = self.Sort(datasetandLabel, index)
        #print(dataset)
        #print(label)
        #theta表示阈值, 小于阈值时停止迭代
        #dataset已经从小到大被排序好
        #计算label的类别，k = dataset.shape[0]/ 3 * class

        #k从2开始分类，分类完成后检查信息熵大小
        k_index = []
        center_location = []
        #首先设置无限大的质心距离
        distans = float('inf')
        count = 0#记录下面循环中k的数目

        for i in range(label.shape[0]-1):
            if label[i] != label[i+1] and count == 0:
                if dataset[i][index] == dataset[i+1][index]:
                    k_index.append(i)
                    center_location.append(dataset[i][index])
                    count += 1
                else:
                    k_index.append(i)
                    center_location.append(dataset[i][index])
                    k_index.append(i + 1)
                    center_location.append(dataset[i + 1][index])
                    count += 2
            elif label[i] != label[i+1] and dataset[i][index] != dataset[i+1][index]:
                k_index.append(i+1)
                center_location.append(dataset[i + 1][index])
                count += 1
            if count == k:
                break
        center_location = np.array(center_location, dtype=float)
        #print('count:', count)
        #print('k:', k)
        #print('label:', label)
        #print('data:', dataset[:, index])
        #print('center_location:', center_location)
        all_iter = 0
        location_history = []
        while distans > theta and all_iter < 100:
            location_history.append(center_location)
            all_iter += 1
            k_cluster = {}  # 每个k聚类的群体
            label_cluster = {} # 每次聚类的信息熵
            #还需要确定聚类后能否合并相邻区域
            for i in range(label.shape[0]):
                total = np.abs(center_location - float(dataset[i][index]))
                argmin = np.argmin(total)
                if center_location[argmin] not in k_cluster.keys():
                    k_cluster[center_location[argmin]] = []
                    label_cluster[center_location[argmin]] = []
                k_cluster[center_location[argmin]].append(float(dataset[i][index]))
                label_cluster[center_location[argmin]].append(label[i])
            k_shannon = shannon
            sig_label = []
            for key in label_cluster:
                sig_label.append(self.maxinlist(label_cluster[key]))
            #print(sig_label)
            #print(k_cluster)
            #print(label_cluster)
            newcenter_location = []
            sig_label_iter = 0
            while sig_label_iter < len(sig_label)-1:
                if sig_label[sig_label_iter] == sig_label[sig_label_iter+1]:
                    #print('keyword:', center_location[sig_label_iter+1])
                    newcenter_location.append(
                        k_cluster[center_location[sig_label_iter+1]][0] / 2 + k_cluster[center_location[sig_label_iter]][
                            -1] / 2)
                    k_cluster[center_location[sig_label_iter]].extend(k_cluster[center_location[sig_label_iter+1]])
                    label_cluster[center_location[sig_label_iter]].extend(label_cluster[center_location[sig_label_iter+1]])
                    del k_cluster[center_location[sig_label_iter+1]]
                    del label_cluster[center_location[sig_label_iter+1]]
                    center_location = np.delete(center_location, sig_label_iter+1)
                    del sig_label[sig_label_iter+1]
                    sig_label_iter -= 1
                else:
                    newcenter_location.append(center_location[sig_label_iter])
                    if sig_label_iter + 1 == len(sig_label) - 1:
                        newcenter_location.append(center_location[-1])
                sig_label_iter += 1
            if len(newcenter_location) != 0:
                center_location = newcenter_location
            '''
            iter_k = 0
            new_k_cluster = {}
            for key in k_cluster:
                new_k_cluster[center_location[iter_k]] = k_cluster[key]
                iter_k += 1
            k_cluster = new_k_cluster
            '''

            #print('newcenter1:', newcenter_location)
            #print(center_location)
            #print(label_cluster)
            #print('k_cluster:', k_cluster)
            #print('center_location:', center_location)
            for key in label_cluster:
                k_shannon -= self.ShannonEnt(label_cluster[key]) * len(label_cluster[key]) / len(label)
            k_shannons.append(k_shannon)
            #print('k_shannons: ', k_shannons)
            # 重新计算中心值
            newcenter_location = np.zeros(len(center_location))

            count = 0
            for key in k_cluster:
                #print(k_cluster[key])
                mean = np.array(k_cluster[key]).mean()
                #print(mean)
                newcenter_location[count] = mean
                count += 1
            distans = np.sum(np.abs(center_location - newcenter_location))/len(center_location)
            #print("distans:", distans)
            center_location = newcenter_location
        maxarg = k_shannons.index(max(k_shannons))
        #print('history shannons:', k_shannons[maxarg])
        center_location = location_history[maxarg]
        #print('history center:', center_location)
        k_cluster = {}  # 每个k聚类的群体
        label_cluster = {}  # 每次聚类的标签，用于计算信息熵和重心
        index_cluster = {} #每个聚类的
        # 还需要确定聚类后能否合并相邻区域
        for i in range(label.shape[0]):
            total = np.abs(center_location - float(dataset[i][index]))
            argmin = np.argmin(total)
            if center_location[argmin] not in k_cluster.keys():
                k_cluster[center_location[argmin]] = []
                label_cluster[center_location[argmin]] = []
                index_cluster[center_location[argmin]] = []
            k_cluster[center_location[argmin]].append(float(dataset[i][index]))
            label_cluster[center_location[argmin]].append(label[i])
            index_cluster[center_location[argmin]].append(i)
        sig_label = []
        for key in label_cluster:
            sig_label.append(self.maxinlist(label_cluster[key]))
        #print(sig_label)
        newcenter_location = []
        sig_label_iter = 0
        while sig_label_iter < len(sig_label) - 1:
            if sig_label[sig_label_iter] == sig_label[sig_label_iter + 1]:
                # print('keyword:', center_location[sig_label_iter+1])
                newcenter_location.append(
                    k_cluster[center_location[sig_label_iter + 1]][0] / 2 + k_cluster[center_location[sig_label_iter]][
                        -1] / 2)
                k_cluster[center_location[sig_label_iter]].extend(k_cluster[center_location[sig_label_iter + 1]])
                label_cluster[center_location[sig_label_iter]].extend(
                    label_cluster[center_location[sig_label_iter + 1]])
                del k_cluster[center_location[sig_label_iter + 1]]
                del label_cluster[center_location[sig_label_iter + 1]]
                center_location = np.delete(center_location, sig_label_iter + 1)
                del sig_label[sig_label_iter + 1]
                sig_label_iter -= 1
            else:
                newcenter_location.append(center_location[sig_label_iter])
                if sig_label_iter + 1 == len(sig_label) - 1:
                    newcenter_location.append(center_location[-1])
            sig_label_iter += 1
        if len(newcenter_location) != 0:
            center_location = newcenter_location
        #print('last index:', index_cluster)
        #print('last label:', label_cluster)
        return center_location, index_cluster, label_cluster, k_shannons[maxarg]
    def K_means_continue(self, dataset, label, index, theta):
        classes = []
        for i in range(label.shape[0]):
            if label[i] not in classes:
                classes.append(label[i])
        max_k = dataset.shape[0] / (3 * len(classes))
        print(max_k)
        max_k = int(max_k) + 1
        k = 2
        if max_k <= 2:
            max_k = k
        maxshonnon = 0
        k_maxshonnon = 2
        while k <= max_k:
            c, i, l, s = self.K_means(dataset, label, index, theta, k)
            if s > maxshonnon:
                maxshonnon = s
                k_maxshonnon = k
            k += 1
        c, i, l, s = self.K_means(dataset, label, index, theta, k_maxshonnon)
        return c, s
def DivideData(dataset, label, train=0.8, valid=None):
    lens = dataset.shape[0]
    DataSetandLabel = np.column_stack((dataset, label))
    train_lens = int(train * lens)
    sample_list = [i for i in range(lens)]
    if valid == None:
        #random_index = random.sample(sample_list, train_lens)
        random_index = [i for i in range(train_lens)]
        #print(random_index)
        train_set = []
        test_set = []
        for i in range(lens):
            if i in random_index:
                train_set.append(DataSetandLabel[i])
            else:
                test_set.append(DataSetandLabel[i])
        return np.array(train_set), np.array(test_set), None
    else:
        valid_lens = int(valid * lens)
        random_index_t = random.sample(sample_list, train_lens)
        random_index_v = random.sample(random_index_t, valid_lens)
        train_set = []
        valid_set = []
        test_set = []
        for i in range(lens):
            if i in random_index_v:
                valid_set.append(DataSetandLabel[i])
            elif i in random_index_t:
                train_set.append(DataSetandLabel[i])
            else:
                test_set.append(DataSetandLabel[i])
        return np.array(train_set), np.array(test_set), np.array(valid_set)

if __name__ == '__main__':
    #西瓜
    #dataset = getDataSetinFile('watermelon.txt')
    #dataset, label = createDataSet(dataset)
    # print(dataset)
    #character = getdiscreteflag(dataset, [6, 7])
    dataset = np.load('iris_dataset.npy')
    label = np.load('iris_label.npy')
    charaName = np.load('iris_feature.npy')
    character = getdiscreteflag(dataset, np.arange(0,4))
    train_data, test_data, _ = DivideData(dataset, label)
    train_dataset = train_data[:, :-1]
    train_label = train_data[:, -1]

    test_dataset = test_data[:, :-1]
    test_label = test_data[:, -1]
    root = DicisionTree(dataset=train_dataset, label=train_label, character=character, flag=False, theta=0.1, KMean=False)
    # root = DicisionTree(dataset, label, flag=True, theta=0.01)
    # print('printing')
    #
    #charaName = np.array(['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率'])
    root.Nameresult(charaName)
    # root.Beprune(dataset, label)
    # root.PrintTree()
    prc = ShowTree.plotTree(root)
    count = 0
    for i in range(len(test_label)):
        result = root.Judge(test_dataset[i])
        if result == test_label[i]:
            count += 1
    print('NoK accuracy: ', count/len(test_label))
    root = DicisionTree(dataset=train_dataset, label=train_label, character=character, flag=False, theta=0,
                        KMean=True)
    root.Nameresult(charaName)
    # root.Beprune(dataset, label)
    # root.PrintTree()
    prc = ShowTree.plotTree(root)
    count = 0
    for i in range(len(test_label)):
        result = root.Judge(test_dataset[i])
        if result == test_label[i]:
            count += 1
    print('Kmean accuracy: ', count / len(test_label))
    # root.Beprune(dataset, label)
    # root.PrintTree()
