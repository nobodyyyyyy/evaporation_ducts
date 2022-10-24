#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import random
import joblib
import sqlite3
from .algorithm import *
from concurrent.futures import ThreadPoolExecutor, as_completed
from .connect_mysql import *
import datetime
import numpy as np
import torch
import sqlite3
import sys
sys.path.append('..')
from Ensembel.Ensemble import Ensemble_Method
from PredictionMethod import GRU, LSTM, RF, SVR, TCN
from PredictionMethod.Measure import GetMAP, GetMAPE, GetRMSE, R_square
from Ensembel.utils import readfile, Divide, TrainValidTest, Ones
from threading import Thread,RLock
import pandas as pd
device = torch.device("cpu")
#模型预测RMSE的阈值
RMSE=2.0
#在预测时加锁防止重新训练时修改模型
lock = RLock()
TrainFlag = False
#判断字符串是否为数字
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

#定义为用于Online的模型
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
        self.Models=[]
        self.epoch = 100
        self.batch = 100
        self.weight = []
        self.max = None
        self.min = None


    def Model_Select(self):
        #一共有5个集成模型，根据校验集选取最好的模型
        train_x, train_y, valid_x, valid_y = self.data_process()
        # 通过训练集和验证集，在模型集合中计算出各个模型的权重，准确度越高，权重越大
        self.weight, _ = self.Weight_rate(self.Models, train_x, train_y, valid_x, valid_y)
        # print(self.weight)
        return

    def setMax_Min(self):
        _, x_min, x_max = Ones(self.data, 1)
        self.max = x_max
        self.min = x_min

    # 归一化处理
    def data_process(self):
        x, x_min, x_max = Ones(self.data, 1)
        self.max = x_max
        self.min = x_min
        y = x
        hist_x, fore_y = Divide(x, y, his_step=self.his_step, pre_step=self.pre_step)
        train_x, train_y, valid_x, valid_y = TrainValidTest(hist_x, fore_y)
        return train_x, train_y, valid_x, valid_y

    # 该方法用于选择在self.Models中，效果最好，也就是权重最大的模型进行越策
    def BestModel_Prediction(self, hist_x):
        # 实际数据归一化
        hist_x = (hist_x - self.min) / (self.max-self.min)
        # ----
        # 选取最合适模型
        # best_index = np.argmax(np.array(self.weight))
        # Model = self.Models[best_index]
        # ----
        #加载模型
        if(os.path.exists("./model/model.pt")):
            Model=torch.load('./model/model.pt')
        else:
            Model = joblib.load('./model/model.model')
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



# 在运算结果超出某一个阈值后，重新训练模型
def Re_Train(retrain_args):
    OM = retrain_args[0]
    data = retrain_args[1]
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
    #更新old模型
    for i in range(len(OM.Models)):
        if(OM.Models[i].name=='LSTM'):
            torch.save(OM.Models[i], './model/old_lstm.pt')
        elif(OM.Models[i].name=='GRU'):
            torch.save(OM.Models[i], './model/old_gru.pt')
        elif (OM.Models[i].name == 'TCN'):
            torch.save(OM.Models[i], './model/old_tcn.pt')
        elif (OM.Models[i].name == 'RF'):
            joblib.dump(OM.Models[i], './model/old_rf.model')
        elif (OM.Models[i].name == 'SVR'):
            joblib.dump(OM.Models[i], './model/old_svr.model')
    best_index = np.argmax(np.array(OM.weight))
    Model = OM.Models[best_index]
    #更新最好的model
    if Model.name=='LSTM' or Model.name=='GRU' or Model.name=='TCN':
        torch.save(Model, './model/model.pt')
        if (os.path.exists("./model/model.model")):
            os.remove('./model/model.model')
    else:
        joblib.dump(Model,'./model/model.model')
        if (os.path.exists("./model/model.pt")):
            os.remove('./model/model.pt')
    global TrainFlag
    TrainFlag = False

# 使用sqlalchemy与后端数据库进行连接

import threading
# generate temperature, humidity, pressure, wind, direction, time, position
class Generation():
    # 缓存中的基本气象数据信息<300
    database = []
    # 缓存中的蒸发波导高度信息
    evap_duct = []

    # 插值算法的参数 ----------------------------
    # 插值起始高度
    # 插值结束高度
    end = 3
    # 插值个数
    nums = 299
    # 插值方法
    kind = "cubic"
    # ------------------------------------------
    #正在训练的标识                                                       
    # socket
    # 用来进行前后端的连接
    socketio = None

    # 为了在该界面保存蒸发波导和表面悬空波导信息
    # 需要在class中存放高度信息height和大气折射率refraction
    # 历史最近的高度信息和大气折射率将存放在npy文件中
    def __init__(self, sqlconnect, OM, cursor, socketio=None, limit=3000):
        # 模型训练用的线程池初始化
        self.trainThread = ThreadPoolExecutor(max_workers=1)
        # 初始化主线程中的socketio相关信息
        self.socketio = socketio
        # 初始化查询文件的位置
        #self.filepath = filepath
        # 初始化插值高度限制
        self.limit = limit
        # 传入与mysql进行交互的对象
        self.sqlconnect = sqlconnect
        self.OM = OM
        #连接数据库操作初始化
        self.cursor = cursor
        # 用limit限制数据长度
        self.cursor.execute("SELECT Time, Temperature1, Humidity1, TemperatureSeaIR, Pressure, WinspeedR1, WindirectR1, temp1 from 实时数据 ORDER BY Time DESC LIMIT " + str(limit) + ";")
        
        #初始船上数据库记录时间

        self.result = self.cursor.fetchall()
        self.result.reverse()
        self.start_time = self.result[-1][0]

        saveInMysql = False
        finalTimeInMysql = self.sqlconnect.Query_FinalTime()
        for i in range(len(self.result)):
            if self.result[i][0] != None and self.result[i][1] != None and self.result[i][2] != None and \
               self.result[i][3] != None and self.result[i][4] != None and self.result[i][5] != None and\
               self.result[i][6] != None:
                evap_val = self.generate_evap(i)
                evap_val = round(evap_val, 2)
                if i != len(self.result) - 1:
                    futu_val = self.generate_future_rand(i)
                else:
                    futu_val = 0
                temp = {
                    'tem': self.result[i][1],
                    'wind': self.result[i][5],
                    'press': self.result[i][4],
                    'direction': self.result[i][6],
                    'hum': self.result[i][2],
                    'times': self.result[i][0],
                    'lat_lon': self.result[i][7],
                }
                evap = {
                    'evap': evap_val,
                    'futu': futu_val,
                    'times': self.result[i][0],
                    'lat_lon': self.result[i][7],
                }
                # 将数据库中没有的数据从db3中存入mysql
                if saveInMysql or self.result[i][0] > finalTimeInMysql:
                    saveInMysql = True
                    self.sqlconnect.Update_Basic(times=datetime.datetime.strptime(temp['times'], "%Y-%m-%d %H:%M:%S"), temperature=temp['tem'], humidity=temp['hum'],
                                             press=temp['press'], direction=temp['direction'], wind=temp['wind'])
                    self.sqlconnect.Update_Evap(times=datetime.datetime.strptime(evap['times'], "%Y-%m-%d %H:%M:%S"), evap_height=evap['evap'], futu_height=evap['futu'], lat_lon=evap['lat_lon'])
                self.evap_duct.append(evap)
                self.database.append(temp)

    def generate_interpolation_data(self, dataset):
        data = np.array(dataset)
        ref, h = generate_data(data)
        return ref, h

    #随机初始化未来波导
    def generate_future_rand(self, i):
        future = zhengfaMs(self.result[i+1][1], self.result[i+1][3], self.result[i+1][2], self.result[i+1][5], self.result[i+1][4])
        return float(future)

    # 这里用来获取近地面的气象数据信息
    # 并将这些气象数据用来更新波导信息
    def generate_base(self):
        # dataset = self.ReadTxt(self.filepath, self.limit)
        dataset = self.ReadGaokong1(self.filepath, self.limit)
        # _, tem, press, wind, hum, direction = dataset[0]
        # times = time.strftime("%Y-%m-%d, %H:%M")
        return dataset

    def generate_evap(self, i):
        evap = zhengfaMs(self.result[i][1], self.result[i][3], self.result[i][2], self.result[i][5], self.result[i][4])
        return float(evap)

    # 返回当前evap处的预测值，并将evap与上一时间点比较计算rmse
    def generate_future(self, evap):
        # 选取evap_duct中最近的5条数据和evap结合进行预测
        evap_for_pred = []
        len_evap = len(self.evap_duct)
        for i in range(0,5):
            evap_for_pred.append(self.evap_duct[len_evap-5+i]['evap'])
        evap_for_pred.append(evap)
        futu = self.OM.BestModel_Prediction(evap_for_pred)
        #计算evap_duct中最近的预测值与evap的rmse误差
        rmse=GetRMSE(np.array(evap), np.array(self.evap_duct[len_evap-1]['futu']))
        rmse=float(rmse)
        futu=float(futu)
        return futu,rmse

    # 仅在update更新时进行调用
    # 凭借多线程的底层进行处理

    # 使用socketio发送给index页面和evaporation页面
    def send_general(self, temp, evap):
        temp['evap'] = evap['evap']
        self.socketio.emit('server_response',
                      {'data': temp}, namespace='/test_conn')


    def send_future(self, temp, evap):
        temp['evap'] = evap['evap']
        temp['futu'] = evap['futu']
        self.socketio.emit('server_response',
                      {'futu': temp}, namespace="/future")


    # 具有多线程协作的update方法
    def update(self):
        print("开始更新！")
        temp = {
            'tem': 0,
            'wind': 0,
            'press': 0,
            'direction': 0,
            'hum': 0,
            'times': 0,
            'sst': 0,
            'lat_lon': "数据缺失"
        }
        #读取船载数据库的气象信息
        self.cursor.execute("SELECT Time, Temperature1, Humidity1, TemperatureSeaIR, Pressure, WinspeedR1, WindirectR1, temp1 from 实时数据 ORDER BY Time DESC LIMIT 1;")
        result = self.cursor.fetchall()[0]
        evap_time = result[0]
        evap_temperature = result[1]
        evap_humidity = result[2]
        evap_sst = result[3]
        evap_pressure = result[4]
        evap_windspeed = result[5]
        evap_winddire = result[6]
        evap_latlon = result[7]

        print("evap time: ", evap_time)
        print("self.start_time: ", self.start_time)
        if evap_time != None and evap_temperature != None and evap_humidity != None and evap_sst != None and evap_pressure != None and evap_windspeed != None :
            # 为了方便，先用 "==" 取代 "!="
            if evap_time == self.start_time:
                print("蒸发波导更新: " + evap_time)
                self.start_time = evap_time
                temp['tem'] = evap_temperature
                temp['wind'] = evap_windspeed
                temp['press'] = evap_pressure
                temp['direction'] = evap_winddire
                temp['hum'] = evap_humidity
                temp['times'] = evap_time
                temp['sst'] = evap_sst
                temp['lat_lon'] = evap_latlon
                # 向数据库更新基本气象数据信息
                evap_val = zhengfaMs(evap_temperature, evap_sst, evap_humidity, evap_windspeed, evap_pressure)
                evap_val = round(evap_val, 2)
                self.sqlconnect.Update_Basic(times=datetime.datetime.strptime(evap_time, "%Y-%m-%d %H:%M:%S"), temperature=temp['tem'], press=temp['press'], wind=temp['wind'],
                                             humidity=temp['hum'], direction=temp['direction'], sst=temp['sst'])
                 #在预测时加锁
                lock.acquire()
                futu_val, rmse = self.generate_future(evap_val)
                lock.release()
                # 如果RMSE超过阈值，创建一个线程利用缓存中的evap_duct对OM重新训练
                global TrainFlag
                print("futu val: ", futu_val)
                print("RMSE: ", rmse)
                print("TrainFlag: ", TrainFlag)
                if rmse > RMSE and not TrainFlag:
                    TrainFlag = True
                    evap_vals = []
                    for j in range(len(self.evap_duct)):
                        evap_vals.append(self.evap_duct[j]['evap'])
                    evap_vals.append(evap_val)
                    retrain_args = [self.OM, evap_vals]
                    print("start train......")
                    testTrain = self.trainThread.submit(Re_Train, retrain_args)
                self.sqlconnect.Update_Evap(times=datetime.datetime.strptime(evap_time, "%Y-%m-%d %H:%M:%S"),
                                            evap_height=evap_val, futu_height=futu_val, lat_lon=evap_latlon)
                evap = {
                    'times': evap_time,
                    'evap': evap_val,
                    'futu': futu_val,
                    'lat_lon': evap_latlon,
                }
                # 服务器推送给前端
                self.send_general(temp, evap)
                self.send_future(temp, evap)
                self.evap_duct.append(evap)
                self.database.append(temp)