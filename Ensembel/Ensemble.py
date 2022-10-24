#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
import copy
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch.autograd import Variable
# import sys
# sys.path.append('..')
from Ensembel.utils import readfile, Divide, TrainValidTest, Ones
from PredictionMethod.Measure import GetMAP, GetMAPE, GetRMSE, R_square
from CARTMethod.CART_XG import caRtXGTree
from CARTMethod.CART_R import caRtTree
import random
# from sko.PSO import PSO

from PredictionMethod import GRU, LSTM, RF, SVR, TCN


device = torch.device("cpu")
cuda_gpu = torch.cuda.is_available()
class Ensemble_Method():
    def __init__(self, data, his_step=6, pre_step=1, input_size=1, output_size=1, min_sample_splite=2):
        '''
        :param select: w->weight, a->adaboost, x->xgboost
        :param his_step: feature
        :param pre_step: result
        :param input_size: input data dim
        :param output_size: output data dim
        :param min_sample_splite: num sample
        '''
        self.data = data
        self.his_step = his_step
        self.pre_step = pre_step
        self.input_size = input_size
        self.output_size = output_size
        self.min_sample_splite = min_sample_splite

        # 模型方法
        self.select = None
        self.BestModelSet = None

        #SVR模型
        svr = SVR.Svr()
        #LSTM
        hidden_size = 12
        lstm = LSTM.LSTM(input_size=input_size, output_size=output_size, hidden_size=hidden_size)
        #GRU
        hidden_size = 12
        gru = GRU.GRU(input_size=input_size, output_size=output_size, hidden_size=hidden_size)
        #RF
        max_depth = 10
        num = 300
        rf = RF.RF(max_depth=max_depth, num=num, min_split=self.min_sample_splite)
        #TCN
        num_channels = [64, 64, 64]
        kernels = 2
        tcn = TCN.TCN(input_size=1, output_size=1, num_channels=num_channels, kernel_size=kernels, dropout=0)
        self.Models = [gru, tcn, lstm, svr, rf]
        self.epoch = 100
        self.batch = 50

    def Model_Select(self):
        # 一共有5种集成方法，根据校验集选取最好的模型
        # 分别是weight、best、adaboost、xgboost、xgbost_multi
        R2 = []
        Regressions = []
        parameter = []
        method_name = ['weight', 'best', 'adaboost', 'xgboost', 'xgboost_multi']
        M = None
        # Weight_Method
        r2, regressions, weight = self.Weight_Method()
        R2.append(r2)
        Regressions.append(regressions)
        parameter.append(weight)
        # best_Method
        r2, regressions = self.BestModel()
        R2.append(r2)
        Regressions.append(regressions)
        # adaboost
        r2, regressions = self.Adaboost(Iter=5)
        R2.append(r2)
        Regressions.append(regressions)
        # xgboost optimization
        # pso = PSO(func=self.OP_XGboost, dim=3, pop=10, max_iter=10, lb=[0.1, 1, 0], ub=[0.5, 20, 2], c1=0.5, c2=0.5, w=0.7)
        # pso.run()
        # best_parameter = pso.gbest_x
        r2, regressions = self.XGboost(epsilon=0.2, lamda=1, gamma=1, max_depth=100)
        R2.append(r2)
        Regressions.append(regressions)
        parameter.append(0.2)
        # xgboost_multi optimization
        # pso = PSO(func=self.OP_XGboost_M, dim=1, pop=10, max_iter=10, lb=[0.1], ub=[0.5], c1=0.5,
                  # c2=0.5, w=0.7)
        # pso.run()
        # best_parameter = pso.gbest_x
        r2, regressions, M = self.XGboost_multi(epsilon=0.2)
        R2.append(r2)
        parameter.append(0.2)
        print(r2)
        max_r2 = np.argmax(np.array(R2))
        self.select = method_name[max_r2]
        self.BestModelSet = Regressions[max_r2]
        return parameter, M

    def Predict(self, x, parameter, M):
        x, x_min, x_max = Ones(x, 1)
        y = x
        hist_x, fore_y = Divide(x, y, his_step=self.his_step, pre_step=self.pre_step)
        if self.select == 'weight':
            print('weight')
            result_final = self.Weight_Method_Prediction(hist_x, self.BestModelSet, parameter)

        elif self.select == 'best':
            print('best')
            result_final = self.BestModel_Prediction(hist_x, self.BestModelSet)

        elif self.select == 'adaboost':
            print('adaboost')
            result_final = self.Adaboost_Prediction(hist_x, self.BestModelSet)

        elif self.select == 'xgboost':
            print('xgboost')
            result_final = self.XGboost_Prediction(hist_x, self.BestModelSet, parameter)

        elif self.select == 'xgboost_multi':
            print('xgboost_multi')
            result_final = self.XGboost_M_Prediction(hist_x, self.BestModelSet, parameter, M)

        result_final = result_final * (x_max - x_min) + x_min
        test_y = fore_y * (x_max - x_min) + x_min
        print('final result in my prediction method.')
        print('RMSE: ', GetRMSE(test_y, result_final))
        print('MAPE: ', GetMAPE(test_y, result_final))
        print('MAP: ', GetMAP(test_y, result_final))
        print('R2: ', R_square(test_y, result_final))
        return result_final

    def OP_XGboost(self, e, l, g):
        R, G = self.XGboost(Iter=10, epsilon=e, lamda=l, gamma=g)
        return -R

    def OP_XGboost_M(self, e):
        R, G, M = self.XGboost_multi(Iter=10, epsilon=e)
        return -R
    # 神经网络模型训练
    def Train(self, model, train_x, train_y, D=None):
        criterion = nn.MSELoss()
        print('name: ', model.name)
        the_lr = 1e-3
        the_decay = 0
        model = model.train()
        if model.name == 'TCN':
            the_lr = 1e-3
        elif model.name == 'LSTM':
            the_lr = 1e-2
            # the_decay = 1e-6
        elif model.name == 'GRU':
            the_lr = 1e-2
            # the_decay = 1e-6
        optimizer = torch.optim.Adam(model.parameters(), lr=the_lr, weight_decay=the_decay)
        train_loader = DataLoader(dataset=TensorDataset(train_x, train_y), batch_size=self.batch, shuffle=False)
        if cuda_gpu:
            model = model.to(device)
            criterion = criterion.to(device)
        for e in range(self.epoch):
            for i, data in enumerate(train_loader):
                inputs, target = data
                if cuda_gpu:
                    inputs, target = Variable(inputs).to(device), Variable(target).to(device)
                else:
                    inputs, target = Variable(inputs), Variable(target)

                if D is not None:
                    if len(D[i*self.batch:(i+1) * self.batch]) == self.batch:
                        D_batch = len(D) * np.sum(D[i*self.batch:(i+1) * self.batch]) / self.batch
                    else:
                        D_batch = len(D) * np.sum(D[i*self.batch:(i+1) * self.batch]) / len(D[i*self.batch:(i+1) * self.batch])
               	else:
                    D_batch = 1
                #print('D_batch: ', D_batch)
                if model.name == 'TCN':
                    y_hat = model(inputs)
                elif model.name == 'LSTM':
                    y_hat, _ = model(inputs, None)
                elif model.name == 'GRU':
                    y_hat, _ = model(inputs, None)
                #print('D_batch:', D_batch)
                loss = criterion(y_hat, target) * D_batch
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if (e + 1) % 10 == 0:
                print('Epoch [{}/{}], Loss:{:.4f}'.format(e + 1, self.epoch, loss.item()))
        return model
    def Valid_Model(self, Model, valid_x, valid_y):
        R2_total = []
        for i in range(len(Model)):
            print(Model[i].name)
            if Model[i].name == 'LSTM' or Model[i].name == 'GRU':
                result, _ = Model[i](valid_x.view(-1, self.his_step, self.input_size).to(device), None)
                R2 = R_square(valid_y.detach().numpy(), result.cpu().detach().numpy())
            elif Model[i].name == 'TCN':
                result = Model[i](valid_x.view(-1, self.his_step, self.input_size).to(device))
                R2 = R_square(valid_y.detach().numpy(), result.cpu().detach().numpy())
            else:
                result = Model[i](valid_x)
                R2 = R_square(valid_y.detach().numpy(), result)
            R2_total.append(R2)
        weight = np.array(R2_total)
        weight[weight < 0.5] = 0

        print('R2: ', weight)
        weight = weight / np.sum(weight)
        return weight, R2_total

    def Weight_rate(self, Model, train_x, train_y, valid_x, valid_y):
        for i in range(len(Model)):
            if Model[i].type == 'NN':
                Model[i] = self.Train(Model[i], train_x.view(-1, self.his_step, self.input_size), train_y.view(-1, self.pre_step, self.output_size))
                Model[i] = Model[i].eval()
        #self.gru = self.Train(self.gru, train_x.view(-1, self.his_step, self.input_size), train_y.view(-1, self.pre_step, self.output_size))
        #self.tcn = self.Train(self.tcn, train_x.view(-1, self.his_step, self.input_size), train_y.view(-1, self.pre_step, self.output_size))
        #self.lstm = self.Train(self.lstm, train_x.view(-1, self.his_step, self.input_size), train_y.view(-1, self.pre_step, self.output_size))
            else:
                Model[i].Train(train_x, train_y)
	    #self.rf.Train(train_x, train_y)
        #self.svr.Train(train_x, train_y)

        weight, R2_total = self.Valid_Model(Model, valid_x, valid_y)

        return weight, R2_total

    def Weight_Method(self):
        print('weight method')
        x, x_min, x_max = Ones(self.data, 1)
        y = x
        Model = []
        for m in self.Models:
            if m.name == 'TCN':
                torch.save(m, 'TCN.pth')
                Model.append(torch.load('TCN.pth'))
            else:
                Model.append(copy.deepcopy(m))

        hist_x, fore_y = Divide(x, y, his_step=self.his_step, pre_step=self.pre_step)
        train_x, train_y, valid_x, valid_y, test_x, test_y = TrainValidTest(hist_x, fore_y, valid=0.1)
        weight, _ = self.Weight_rate(Model, train_x, train_y, valid_x, valid_y)
        result_final = []
#        print(1)

        for i in range(len(Model)):
            if Model[i].name == 'LSTM' or Model[i].name == 'GRU':
                result, _ = Model[i](test_x.view(-1, self.his_step, self.input_size).to(device), None)
                result = result.cpu().detach().numpy().ravel()
            elif Model[i].name == 'TCN':
                result = Model[i](test_x.view(-1, self.his_step, self.input_size).to(device))
                result = result.cpu().detach().numpy().ravel()
            else:
                result = Model[i](test_x)
            result_final.append(result)
#        print(2)
        result_final = np.array(result_final)
        print(result_final.shape)
        result_final = result_final * weight.reshape(-1, 1)
        result_final = np.sum(result_final, axis=0)
        result_final = result_final * (x_max - x_min) + x_min
        test_y = test_y.detach().numpy() * (x_max - x_min) + x_min
        print('RMSE: ', GetRMSE(test_y, result_final))
        print('MAPE: ', GetMAPE(test_y, result_final))
        print('MAP: ', GetMAP(test_y, result_final))
        print('R2: ', R_square(test_y, result_final))
        return R_square(test_y, result_final), Model, weight.reshape(-1, 1)

    def Weight_Method_Prediction(self, hist_x, Model=None, weight=[]):
        if Model is None:
            Model = self.Models
        if len(weight) == 0:
            return -999
        result_final = []
        for i in range(len(Model)):
            if Model[i].name == 'LSTM' or Model[i].name == 'GRU':
                result, _ = Model[i](hist_x.view(-1, self.his_step, self.input_size).to(device), None)
                result = result.cpu().detach().numpy().ravel()
            elif Model[i].name == 'TCN':
                result = Model[i](hist_x.view(-1, self.his_step, self.input_size).to(device))
                result = result.cpu().detach().numpy().ravel()
            else:
                result = Model[i](hist_x)
            result_final.append(result)
        result_final = np.array(result_final)
        print(result_final.shape)
        result_final = result_final * weight.reshape(-1, 1)
        result_final = np.sum(result_final, axis=0)
        result_final = result_final
        return result_final

    def BestModel(self):
        print('bestmodel')
        x, x_min, x_max = Ones(self.data, 1)
        y = x
        Model = []
        for m in self.Models:
            if m.name == 'TCN':
                torch.save(m, 'TCN.pth')
                Model.append(torch.load('TCN.pth'))
            else:
                Model.append(copy.deepcopy(m))

        hist_x, fore_y = Divide(x, y, his_step=self.his_step, pre_step=self.pre_step)
        train_x, train_y, valid_x, valid_y, test_x, test_y = TrainValidTest(hist_x, fore_y, valid=0.1)
        weight, _ = self.Weight_rate(Model, train_x, train_y, valid_x, valid_y)
        index = np.argmax(weight)

        if Model[index].name == 'LSTM' or Model[index].name == 'GRU':
            result_final, _ = Model[index](test_x.view(-1, self.his_step, self.input_size).to(device), None)
            result_final = result_final.cpu().detach().numpy().ravel()
        elif Model[index].name == 'TCN':
            result_final = Model[index](test_x.view(-1, self.his_step, self.input_size).to(device))
            result_final = result_final.cpu().detach().numpy().ravel()
        else:
            result_final = Model[index](test_x)
        result_final = np.array(result_final)
        result_final = result_final
        result_final = result_final * (x_max - x_min) + x_min
        test_y = test_y.detach().numpy() * (x_max - x_min) + x_min
        print('RMSE: ', GetRMSE(test_y, result_final))
        print('MAPE: ', GetMAPE(test_y, result_final))
        print('MAP: ', GetMAP(test_y, result_final))
        print('R2: ', R_square(test_y, result_final))
        return R_square(test_y, result_final), Model[index]

    def BestModel_Prediction(self, hist_x, Model=None):
        if Model is None:
            Model = self.Models
        if Model.name == 'LSTM' or Model.name == 'GRU':
            result_final, _ = Model(hist_x.view(-1, self.his_step, self.input_size).to(device), None)
            result_final = result_final.cpu().detach().numpy().ravel()
        elif Model.name == 'TCN':
            result_final = Model(hist_x.view(-1, self.his_step, self.input_size).to(device))
            result_final = result_final.cpu().detach().numpy().ravel()
        else:
            result_final = Model(hist_x)
        result_final = np.array(result_final)
        return result_final

    def Adaboost(self, Iter):
        x, x_min, x_max = Ones(self.data, 1)
        y = x
        hist_x, fore_y = Divide(x, y, his_step=self.his_step, pre_step=self.pre_step)
        train_x, train_y, valid_x, valid_y, test_x, test_y = TrainValidTest(hist_x, fore_y, valid=0.1)
        D = np.ones(len(train_x)) / len(train_x)
        #选择训练集
        Score = np.ones(len(train_x))
        #弱回归器集合
        G = []
        #弱回归器系数
        alpha = []
        for iter in range(Iter):
            #深拷贝模型组
            Model = []
            for m in self.Models:
                if m.name == 'TCN':
                    torch.save(m, 'TCN.pth')
                    Model.append(torch.load('TCN.pth'))
                else:
                    Model.append(copy.deepcopy(m))
            #尽可能选择分数差的例子进入训练
            results = []
            RMSEs = []
            selected_data = []
            for i in range(len(Score)):
                r = np.random.random()
                if r < Score[i]:
                    selected_data.append(i)
            # 如果出现异常值，保持原来的权重训练
            # 这里可以尝试去除少数异常值，直到数据分布没那么夸张再训练
            if len(selected_data) < int(len(D) * 0.8):
                selected_data = np.arange(0, len(D), 1)

            for m in Model:
                if m.type == 'NN':
                    m = self.Train(m, train_x.view(-1, self.his_step, self.input_size),
                                   train_y.view(-1, self.pre_step, self.output_size), D)
                    if m.name != 'TCN':
                        result, _ = m(train_x.view(-1, self.his_step, self.input_size).to(device), None)
                        result = result.cpu().detach().numpy()
                    else:
                        result = m(train_x.view(-1, self.his_step, self.input_size).to(device))
                        result = result.cpu().detach().numpy()
                    results.append(result)
                else:
                    #预处理D，尽可能选择权重大的数据集进行训练，设置选择的训练集的可能性根据权重决定
                    m.Train(train_x[selected_data], train_y[selected_data])
                    result = m(train_x)
                    results.append(result)
                rmse = GetRMSE(result, train_y.detach().numpy())
                RMSEs.append(rmse)

            RMSEs = np.array(RMSEs)
            # 问题是每次都选择效果最好的弱回归器，贪心算法不一定得到最优解
            g = np.argmin(RMSEs)
            G.append(Model[g])
            # 计算新的权重
            max_error = np.max(np.abs(np.array(results[g]).ravel() - np.array(train_y).ravel()))
            err = np.abs(np.array(results[g]).ravel() - np.array(train_y).ravel()) ** 2 / max_error ** 2
            err = np.sum(D * err)
            alpha_k = err / (1 - err)
            alpha.append(alpha_k)
            D = D * alpha_k ** (1 - err) / np.sum(D * alpha_k ** (1 - err))
            Score = D / np.sum(D)
            Score *= len(Score)

        final_results = []
        for m in Model:
            if m.type == 'NN':
                if m.name == 'TCN':
                    result = m(test_x.view(-1, self.his_step, self.input_size).to(device))
                else:
                    result, _ = m(test_x.view(-1, self.his_step, self.input_size).to(device), None)
                result = result.cpu()
                result = result.detach().numpy().ravel()
            else:
                result = m(test_x).ravel()
            final_results.append(result)
        final_results = np.array(final_results)
        #取每个模型的中位数
        center_results = np.zeros(final_results.shape[1])
        for i in range(final_results.shape[1]):
            center_index = int(len(G) * 0.5)
            thelist = np.sort(final_results[:, i])
            center_results[i] = thelist[center_index]
        test_y = test_y.detach().numpy()
        print('final_result: ', final_results.shape)
        print('RMSE: ', GetRMSE(test_y, center_results))
        print('MAPE: ', GetMAPE(test_y, center_results))
        print('MAP: ', GetMAP(test_y, center_results))
        print('R2: ', R_square(test_y, center_results))
        print('over')
        return R_square(test_y, center_results), G

    def Adaboost_Prediction(self, hist_x, Model):
        final_results = []
        for m in Model:
            if m.type == 'NN':
                if m.name == 'TCN':
                    result = m(hist_x.view(-1, self.his_step, self.input_size).to(device))
                else:
                    result, _ = m(hist_x.view(-1, self.his_step, self.input_size).to(device), None)
                result = result.cpu()
                result = result.detach().numpy().ravel()
            else:
                result = m(hist_x).ravel()
            final_results.append(result)
        final_results = np.array(final_results)
        center_index = int(len(Model) * 0.5)
        thelist = np.sort(final_results)
        return thelist[center_index]

    def feature_sample(self, x):
        num_feature = self.his_step
        num_sample = random.randint(self.min_sample_splite, num_feature - 1)
        index_feature = random.sample([i for i in range(num_feature)], num_sample)
        x = x[:, index_feature]
        return x

    def XGboost(self, Iter=10, epsilon=0.3, lamda=2, gamma=1, max_depth=100):
        '''
        :param x: input
        :param y: output
        :param Iter: num of iteration or num of tree
        :param epsilon: shrink factor
        :param min_sample_splite:
        :return:
        '''
        x = self.data
        y = x
        models = []
        Train_max = [1]
        Train_min = [0]
        y_hat = None
        for i in range(Iter):
            #sample_x = self.feature_sample(x)
            hist_x, fore_y = Divide(x, y, his_step=self.his_step, pre_step=self.pre_step)
            #hist_x = self.feature_sample(hist_x)
            train_x, train_y, test_x, test_y = TrainValidTest(hist_x, fore_y)
            _, _, _, _, test_x, test_y = TrainValidTest(hist_x, fore_y, valid=0.1)
            TY = train_y.detach().numpy().ravel()
            if y_hat is None:
                y_hat = np.zeros(len(train_y))
            character = [False for _ in range(hist_x.shape[1])]
            XCART = caRtXGTree(dataset=train_x.detach().numpy(), label=train_y.detach().numpy(), y_hat=y_hat, character=character, lamda=lamda, gamma=gamma, max_depth=max_depth)
            for j in range(train_x.shape[0]):
                result = XCART.Judge(train_x[j])
                y_hat[j] += result * epsilon
            #y_hat, hat_min, hat_max = Ones(y_hat, 1)
            #train_y, y_min, y_max = Ones(train_y, 1)
            result_final = y_hat
            print('RMSE: ', GetRMSE(TY, result_final))
            print('MAPE: ', GetMAPE(TY, result_final))
            print('MAP: ', GetMAP(TY, result_final))
            print('R2: ', R_square(TY, result_final))
            models.append(XCART)

        result_final = np.zeros(len(test_y))
        for i in range(len(models)):
            for j in range(test_x.shape[0]):
                result_final[j] += models[i].Judge(test_x[j]) * epsilon

        result_final = result_final
        test_y = test_y.detach().numpy()
        print('RMSE: ', GetRMSE(test_y, result_final))
        print('MAPE: ', GetMAPE(test_y, result_final))
        print('MAP: ', GetMAP(test_y, result_final))
        print('R2: ', R_square(test_y, result_final))
        return R_square(test_y, result_final), models

    def XGboost_Prediction(self, hist_x, Model, epsilon):
        result_final = 0
        for i in range(len(Model)):
            result_final += Model[i].Judge(hist_x) * epsilon
        return result_final

    def XGboost_multi(self, Iter=10, epsilon=0.5):
        x, x_min, x_max = Ones(self.data, 1)
        y = x
        hist_x, fore_y = Divide(x, y, his_step=self.his_step, pre_step=self.pre_step)
        train_x, train_y, valid_x, valid_y, test_x, test_y = TrainValidTest(hist_x, fore_y, valid=0.1)
        # 弱回归器集合
        G = []
        # 归一化的集合
        Min = [0]
        Max = [1]
        # best num of G
        R2_best = []
        TY = copy.deepcopy(train_y.detach().numpy().ravel() * (x_max - x_min) + x_min)
        # 弱回归器系数
        y_hat = None
        # 深拷贝模型组
        for iter in range(Iter):
            Model = []
            for m in self.Models:
                if m.name == 'TCN':
                    torch.save(m, 'TCN.pth')
                    Model.append(torch.load('TCN.pth'))
                else:
                    Model.append(copy.deepcopy(m))
            RMSEs = []
            results = []
            for m in Model:
                if m.type == 'NN':
                    m = self.Train(m, train_x.view(-1, self.his_step, self.input_size),
                                   train_y.view(-1, self.pre_step, self.output_size))
                    if m.name != 'TCN':
                        result, _ = m(train_x.view(-1, self.his_step, self.input_size).to(device), None)
                        result = result.cpu().detach().numpy()
                    else:
                        result = m(train_x.view(-1, self.his_step, self.input_size).to(device))
                        result = result.cpu().detach().numpy()
                else:
                    # 预处理D，尽可能选择权重大的数据集进行训练，设置选择的训练集的可能性根据权重决定
                    m.Train(train_x, train_y)
                    result = m(train_x)
                results.append(result)
                rmse = GetRMSE(result, train_y.detach().numpy())
                RMSEs.append(rmse)
            min_index = np.argmin(np.array(RMSEs))
            print('min RMSE: ', RMSEs[min_index])
            G.append(Model[min_index])
            if y_hat is None:
                y_hat = (results[min_index] * (Max[iter]-Min[iter]) + Min[iter]) * epsilon
                #y_hat = results[min_index] * epsilon
            else:
                #y_hat += results[min_index] * epsilon
                y_hat += (results[min_index] * (Max[iter]-Min[iter]) + Min[iter]) * epsilon
            train_y -= torch.tensor(y_hat).view(-1, 1)
            train_y, y_min, y_max = Ones(train_y.detach().numpy().ravel(), 1)
            train_y = torch.tensor(train_y).view(-1, 1)
            Min.append(y_min)
            Max.append(y_max)
            result_final = y_hat * (x_max - x_min) + x_min
            print('RMSE: ', GetRMSE(TY, result_final))
            print('MAPE: ', GetMAPE(TY, result_final))
            print('MAP: ', GetMAP(TY, result_final))
            r2 = R_square(TY, result_final)
            R2_best.append(r2) 
            print('R2: ', R_square(TY, result_final))
            #self.epoch -= int(epsilon * self.epoch) 
        result_final = np.zeros(len(test_y))
        r2_max = np.argmax(np.array(R2_best))
        print('R2-', r2_max, ':', R2_best[r2_max])
        for i in range(r2_max+1):
            if m.type == 'NN':
                if m.name == 'TCN':
                    result = G[i](test_x.view(-1, self.his_step, self.input_size).to(device))
                else:
                    result, _ = G[i](test_x.view(-1, self.his_step, self.input_size).to(device), None)
                result = result.cpu()
                result = result.detach().numpy().ravel()
            else:
                result = G[i](test_x).ravel()
            #result_final += result * epsilon
            result_final += (result * (Max[i] - Min[i]) + Min[i]) * epsilon
        result_final = result_final * (x_max - x_min) + x_min
        test_y = test_y.detach().numpy() * (x_max - x_min) + x_min
        print('RMSE: ', GetRMSE(test_y, result_final))
        print('MAPE: ', GetMAPE(test_y, result_final))
        print('MAP: ', GetMAP(test_y, result_final))
        print('R2: ', R_square(test_y, result_final))
        M = []
        M.extend(Max[:r2_max+1])
        M.extend(Min[:r2_max+1])
        return R_square(test_y, result_final), G[:r2_max+1], M

    def XGboost_M_Prediction(self, hist_x, Model, epsilon, M):
        len_M = len(M)
        Max = M[:int(len_M/2)]
        Min = M[int(len_M/2):]
        result_final = 0
        for i in range(len(Model)):
            if Model[i].type == 'NN':
                if Model[i].name == 'TCN':
                    result = Model[i](hist_x.view(-1, self.his_step, self.input_size).to(device))
                else:
                    result, _ = Model[i](hist_x.view(-1, self.his_step, self.input_size).to(device), None)
                result = result.cpu()
                result = result.detach().numpy().ravel()
            else:
                result = Model[i](hist_x).ravel()
            #result_final += result * epsilon
            result_final += (result * (Max[i] - Min[i]) + Min[i]) * epsilon
        return result_final

if __name__ == '__main__':
    filepath_duct = "../PredictionMethod/data8.txt"
    data = readfile(filepath_duct, 'duct')
    #x, x_min, x_max = Ones(data, 1)
    EM1 = Ensemble_Method(data)
    parameter, M = EM1.Model_Select()
    x = data[int(len(data) * 0.8):]
    EM1.Predict(x, parameter, M)

    print('----------------------------------------------------------')
    #EM1.Weight_Method(x, x)
    #EM2 = Ensemble_Method()
    #EM2.BestModel(data, data)

