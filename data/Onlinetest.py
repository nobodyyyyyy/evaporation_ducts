import numpy as np
import torch
import sys
import time
import matplotlib.pyplot as plt
sys.path.append('..')
from Ensembel.Ensemble import Ensemble_Method
from PredictionMethod import GRU, LSTM, RF, SVR, TCN
from PredictionMethod.Measure import GetMAP, GetMAPE, GetRMSE, R_square
from Ensembel.utils import readfile, Divide, TrainValidTest, Ones
from threading import Thread,RLock
import pandas as pd
device = torch.device("cpu")
#设定的阈值
RMSE=2.0
lock = RLock()
flag=False
# 固定随机种子，使得模型的参数在初始化的时候，每次保持一致
def setup_seed(seed):
    # torck的随机种子seed
    torch.manual_seed(seed)
    # 在cuda上的创建的随机种子
    torch.cuda.manual_seed_all(seed)
    # numpy的随机种子
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 定义为用于Online的模型
class Online_Method(Ensemble_Method):
    def __init__(self, data, his_step=6, pre_step=1, input_size=1, output_size=1, min_sample_splite=2):
        self.data = data
        # 历史步长
        self.his_step = his_step
        # 预测的步长
        self.pre_step = pre_step
        # 输入的维度数
        self.input_size = input_size
        # 输出的维度数
        self.output_size = output_size
        # 最小的样本切割，用来限制随机森林中，在划分特征时，最少需要的特征数
        self.min_sample_splite = min_sample_splite
        # 模型方法

        self.select = None
        self.BestModelSet = None
        # SVR模型
        svr = SVR.Svr()
        # LSTM
        hidden_size = 12
        lstm = LSTM.LSTM(input_size=input_size, output_size=output_size, hidden_size=hidden_size)
        # GRU
        hidden_size = 12
        gru = GRU.GRU(input_size=input_size, output_size=output_size, hidden_size=hidden_size)
        # RF
        max_depth = 10
        num = 300
        rf = RF.RF(max_depth=max_depth, num=num, min_split=self.min_sample_splite)
        # TCN
        num_channels = [64, 64, 64]
        kernels = 2
        tcn = TCN.TCN(input_size=1, output_size=1, num_channels=num_channels, kernel_size=kernels, dropout=0)

        self.Models = [gru, tcn, lstm, svr, rf]
        self.epoch = 100
        self.batch = 100
        self.weight = []
        self.max = None
        self.min = None


    def Model_Select(self):
        #一共有5个集成模型，根据校验集选取最好的模型
        # R2 = []
        # Regressions = []
        # parameter = []
        # method_name = ['weight', 'best', 'adaboost', 'xgboost', 'xgboost_multi']
        # Weight_Method
        # setup_seed(20)
        train_x, train_y, valid_x, valid_y = self.data_process()
        # 通过训练集和验证集，在模型集合中计算出各个模型的权重，准确度越高，权重越大
        self.weight, _ = self.Weight_rate(self.Models, train_x, train_y, valid_x, valid_y)
        # print(self.weight)
        return

    # 归一化处理
    def data_process(self):
        x, x_min, x_max = Ones(self.data, 1)
        self.max = x_max
        self.min = x_min
        y = x
        hist_x, fore_y = Divide(x, y, his_step=self.his_step, pre_step=self.pre_step)
        train_x, train_y, valid_x, valid_y= TrainValidTest(hist_x, fore_y)
        return train_x, train_y, valid_x, valid_y

    # 该方法用于选择在self.Models中，效果最好，也就是权重最大的模型进行越策
    def BestModel_Prediction(self, hist_x):
        # 实际数据归一化
        hist_x = (hist_x - self.min) / (self.max-self.min)
        # ----
        # 选取最合适模型
        best_index = np.argmax(np.array(self.weight))
        Model = self.Models[best_index]
        # ----

        # 下面则是不同的模型对应不同计算方式
        if Model.name == 'LSTM' or Model.name == 'GRU':
            hist_x = torch.tensor(hist_x, dtype=torch.float32).unsqueeze(0)
            result_final, _ = Model(hist_x.view(-1, self.his_step, self.input_size).to(device), None)
            result_final = result_final.cpu().detach().numpy().ravel()
        elif Model.name == 'TCN':
            hist_x = torch.tensor(hist_x, dtype=torch.float32).unsqueeze(0)
            result_final = Model(hist_x.view(-1, self.his_step, self.input_size).to(device))
            result_final = result_final.cpu().detach().numpy().ravel()
        else:
            result_final = Model(hist_x.reshape(1, self.his_step))
        result_final = np.array(result_final) * (self.max-self.min) + self.min
        return result_final

    # 该方法用于选择在self.Models中，按照权重进行累加，效果越好的模型，权重越大，属于该部分的结果占比越大
    # 内部计算思路如上
    def Weight_Method_Prediction(self, hist_x):
        result_final = []
        hist_x = (hist_x - self.min) / (self.max - self.min)
        for i in range(len(self.Models)):
            if self.Models[i].name == 'LSTM' or self.Models[i].name == 'GRU':
                hist = torch.tensor(hist_x,dtype=torch.float32).unsqueeze(0)
                result, _ = self.Models[i](hist.view(-1, self.his_step, self.input_size).to(device), None)
                result = result.cpu().detach().numpy().ravel()
            elif self.Models[i].name == 'TCN':
                hist = torch.tensor(hist_x, dtype=torch.float32).unsqueeze(0)
                result = self.Models[i](hist.view(-1, self.his_step, self.input_size).to(device))
                result = result.cpu().detach().numpy().ravel()
            else:
                result = self.Models[i](hist_x.reshape(1, self.his_step))
            result_final.append(result)
        result_final = np.array(result_final)
        # print("result:", result_final)
        result_final = result_final * self.weight.reshape(-1, 1)
        # print("weights:",self.weight)
        # print("result:", result_final)
        result_final = np.sum(result_final, axis=0) * (self.max-self.min) + self.min
        result_final = result_final
        return result_final

# 在运算结果超出某一个阈值后，重新训练模型
def Re_Train(OM,data):
    print("误差过大需要重新训练")
    # SVR模型
    svr = SVR.Svr()
    # LSTM
    hidden_size = 12
    lstm = LSTM.LSTM(input_size=OM.input_size, output_size=OM.output_size, hidden_size=hidden_size)
    # GRU
    hidden_size = 12
    gru = GRU.GRU(input_size=OM.input_size, output_size=OM.output_size, hidden_size=hidden_size)
    # RF
    max_depth = 10
    num = 300
    rf = RF.RF(max_depth=max_depth, num=num, min_split=OM.min_sample_splite)
    # TCN
    num_channels = [64, 64, 64]
    kernels = 2
    tcn = TCN.TCN(input_size=OM.input_size, output_size=OM.output_size, num_channels=num_channels, kernel_size=kernels, dropout=0)
    new_model = [lstm, tcn, gru, rf, svr]
    OM.data = data
    train_x, train_y, valid_x, valid_y = OM.data_process()
    # 旧模型在新的验证集上的R2指标
    print('old:')
    _, R2_old = OM.Valid_Model(OM.Models, valid_x, valid_y)
    # 新模型在新的验证集上的R2指标
    print('new:')
    _, R2_new = OM.Weight_rate(new_model, train_x, train_y, valid_x, valid_y)
    print('new model:', R2_old)
    print('old model', R2_new)
    new_max = np.argmax(np.array(R2_new))
    old_min = np.argmin(np.array(R2_old))
    # 在新的验证集上，效果最差的老模型被效果最好的新模型取代
    OM.Models[old_min] = new_model[new_max]
    R2_old[old_min] = R2_new[new_max]
    # 计算重组后的新模型集的weighted
    OM.weight = np.array(R2_old)/np.sum(R2_old)

if __name__ =="__main__":
    # 数据集文件
    # filepath_duct = "../PredictionMethod/data8.txt"
    # data = readfile(filepath_duct, 'duct')
    # data=np.loadtxt('traffic.txt',dtype=np.float,delimiter=',')
    # data = np.loadtxt('exchange_rate.txt', dtype=np.float, delimiter=',')
    # data = np.loadtxt('evap.txt', dtype=np.float, delimiter='  ')
    data=pd.read_csv('water.csv',header=None)
    data=data.values
    data=data[:,2]
    # print(len(data))
    # re_data数据集模拟重新训练的数据集，此数据为最近的645条数据
    # 在不断接收新数据的情况下，由部分istory[]和fore组成，类似滑动窗口
    re_date = data[int(len(data) * 0.4):int(len(data) * 0.8)]
    # data_len = len(re_date)
    # fore模拟新获取的波到高度信息
    fore = data[int(len(data) * 0.8):]
    # 使用history数据集训练模型，假象为已有的数据集
    history = data[:int(len(data) * 0.4)]
    RSME_best = []
    RSME_weight = []
    MAP=[]
    MAPE=[]
    OM = Online_Method(history)
    t = Thread(target=Re_Train, args=[OM, re_date])
    # OM.Re_Train(re_date)
    # print("fore len:", len(fore))
    # start = time.time()
    # 计算模型集中各个模型的权重
    OM.Model_Select()
    # end = time.time()
    # print("训练消耗时间:", end-start)
    # start = time.time()
    for i in range(len(fore)):
        if i == 0:
            # 当最开始计算的时候，由最近6条历史数据获得预测值
            inputs = re_date[-6:]
        else:
            # 上一轮的实际值更新到inputs中
            np.append(inputs, fore[i-1])
            # 删除inputs中最久的数据
            np.delete(inputs, 0)
        # 权重集成下的预测结果
        lock.acquire()
        # outputs = OM.Weight_Method_Prediction(inputs)
        outputs = OM.BestModel_Prediction(inputs)
        lock.release()
        RSME_weight.append(GetRMSE(fore[i], outputs))
        MAP.append(GetMAP(fore[i], outputs))
        MAPE.append(GetMAPE(fore[i], outputs))
    # RSME_weight.sort()
    # print(len(RSME_weight))
    # print("RMSE set:",RSME_weight)
    # print(RSME_weight[37])
    #     误差超过阈值
    #     if((GetRMSE(fore[i], outputs)>RMSE)&(flag==False)):
    #         flag=True
    #         t.start()
    before_rmse=np.mean(RSME_weight)
    before_map=np.mean(MAP)
    before_mape=np.mean(MAPE)
    print("Before retrain RMSE:",before_rmse)
    print("Before retrain MAP:", before_map)
    print("Before retrain MAPE:", before_mape)
    Re_Train(OM,re_date)
    RSME_weight=[]
    MAP=[]
    MAPE=[]
    for i in range(len(fore)):
        if i == 0:
                # 当最开始计算的时候，由最近6条历史数据获得预测值
                inputs = re_date[-6:]
        else:
                # 上一轮的实际值更新到inputs中
                np.append(inputs, fore[i - 1])
                # 删除inputs中最久的数据
                np.delete(inputs, 0)
            # 权重集成下的预测结果
        # outputs = OM.Weight_Method_Prediction(inputs)
        outputs = OM.BestModel_Prediction(inputs)
        RSME_weight.append(GetRMSE(fore[i], outputs))
        MAP.append(GetMAP(fore[i], outputs))
        MAPE.append(GetMAPE(fore[i], outputs))
    after_rmse=np.mean(RSME_weight)
    after_map = np.mean(MAP)
    after_mape=np.mean(MAPE)
    print("After retrain RMSE:",after_rmse)
    print("After retrain map:", after_map)
    print("After retrain MAPE:", after_mape)
        # 最佳模型的预测结果
        # outputs = OM.BestModel_Prediction(inputs)
        # RSME_best.append(GetRMSE(fore[i], outputs))
    # end = time.time()
    # print((end-start)/len(fore))
    # print(np.sum(RSME_best)/len(fore))
    # print(np.sum(RSME_weight)/len(fore))
    # plt.figure()
    # x_aixs = np.arange(0, len(fore))
    # plt.plot(x_aixs, RSME_weight, label='weight')
    # plt.plot(x_aixs, RSME_best, label='best')
    # plt.legend()
    # plt.show()
        # print('MAPE: ', GetMAPE(fore[i], outputs))
        # print('MAP: ', GetMAP(fore[i], outputs))
        # print('R2: ', R_square(fore[i], outputs))



