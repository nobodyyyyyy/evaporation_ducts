import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from tkinter import _flatten

from Util.TimeUtil import TimeUtil


class DataSet:
    FEATURE_NPS = 7
    FEATURE_BABIN = 8
    FEATURE_LIULI = 9
    FEATURE_PJ = 10
    FEATURE_WEIZHESHELV = 11

    FEATURE_NAME_MAP = {
        FEATURE_NPS: 'nps',
        FEATURE_BABIN: 'babin',
        FEATURE_LIULI: 'liuli',
        FEATURE_PJ: 'pj',
        FEATURE_WEIZHESHELV: 'wei'
    }

    MODEL_NOT_SPECIFY = 0
    MODEL_LSTM = 1
    MODEL_RNN = 2

    def __init__(self, source='../height_model/merged/stn_54511.xlsx', col=FEATURE_NPS, init_feature=True,
                 start_date='2020-01-01', end_date='2021-12-31', machine_learning=False):
        # todo 加时间截取
        self.raw_data = pd.read_excel(source)
        self.ratio = 0.0
        self.x_train = []
        self.y_train = []
        self.x_val = []
        self.y_val = []
        self.step = 1
        self.machine_learning = machine_learning
        self.specific_model = self.MODEL_NOT_SPECIFY
        self.start_date = start_date
        self.end_date = end_date
        self.extract_data_in_time_range()
        try:
            self.date_index = pd.date_range(start_date, end_date)
        except Exception as e:
            print('DataSet __init__ error: {}'.format(e))
            self.date_index = None
        if init_feature:
            target = np.array(self.raw_data.iloc[:, col].to_list(), dtype=np.float64)
            self.data_scaler = MinMaxScaler()
            self.data = self.data_scaler.fit_transform(target.reshape(-1, 1))

    def extract_data_in_time_range(self):
        """
        截取数据集，拿训练时间 + 后面两天的数据
        :return:
        """
        new_data = []
        start = False
        start_t = TimeUtil.str_to_timestamp(self.start_date)
        end_t = TimeUtil.str_to_timestamp(self.end_date)
        for _, row in self.raw_data.iterrows():
            ts = TimeUtil.str_to_timestamp(row['日期'])
            if not start:
                if ts >= start_t:
                    start = True
            if start:
                new_data.append(row)
                if ts >= end_t:
                    self.raw_data = pd.DataFrame(new_data)
                    return

    def change_feature(self, col=FEATURE_NPS):
        target = np.array(self.raw_data.iloc[:, col].to_list(), dtype=np.float64)
        self.data_scaler = MinMaxScaler()
        self.data = self.data_scaler.fit_transform(target.reshape(-1, 1))

    def inverse_transform(self, data):
        if len(data.shape) == 1:
            data = np.expand_dims(data, 1)
        return self.data_scaler.inverse_transform(data)

    def self_inverse_transform(self):
        return self.inverse_transform(self.data)

    def split(self, input_data, ratio_train=0.8, ratio_val=0.2, ratio_test=0.0, input_window=12, output_window=3,
              machine_learning=False, step=1, specify_model=MODEL_NOT_SPECIFY, web_split=False, web_split_len=2):
        if self.ratio == ratio_train and machine_learning == self.machine_learning and step == self.step \
                and specify_model == self.specific_model:
            return self.x_train, self.y_train, self.x_val, self.y_val
        else:
            return self._split(input_data, ratio_train, ratio_val, ratio_test=ratio_test, input_window=input_window,
                               output_window=output_window,
                               machine_learning=machine_learning, step=step,
                               specify_model=specify_model,
                               web_split=web_split,
                               web_split_len=web_split_len)

    def _split(self, input_data, ratio_train=0.8, ratio_val=0.2, ratio_test=0.0,
               input_window=12, output_window=3,
               machine_learning=False, step=1, specify_model=MODEL_NOT_SPECIFY, web_split=False, web_split_len=2):
        """
        训练测试划分
        :param input_data:
        :param ratio_train:
        :param ratio_val:
        :param ratio_test:
        :param input_window:
        :param output_window: 用于深度学习模型
        :param machine_learning:
        :param step: 和 output_window 互斥，step 的设置主要是让机器学习模型学习跨时间步的预测的
        :return:
        """
        if output_window != 1 and step != 1:
            print('_split... output_window and step can not != 1 at the same time.')
            return [], [], [], []
        self.step = step
        self.ratio = ratio_train
        self.machine_learning = machine_learning
        self.x_train = []
        self.y_train = []
        self.x_val = []
        self.y_val = []

        _len = len(input_data)
        if not web_split:
            _train_len = int(_len * ratio_train)
            _val_len = int(_len * ratio_val)  # val 就是 test
            _test_len = int(_len * (1 - ratio_train - ratio_val))
            for _ in range(0, _train_len - input_window - step + 1):
                self.x_train.append(input_data[_: _ + input_window])
                self.y_train.append(
                    input_data[_ + input_window + step - 1: _ + input_window + step - 1 + output_window])

            self.x_train = np.array(_flatten(self.x_train))
            self.y_train = np.array(_flatten(self.y_train))
            for _ in range(_train_len, _len - input_window - output_window - step + 1):  # todo 起点不对
                self.x_val.append(input_data[_: _ + input_window])
                self.y_val.append(input_data[_ + input_window + step - 1: _ + input_window + step - 1 + output_window])
        else:
            # 网站训练只拿 48 小时去测试
            # 需求更新，支持用户选择天数，24 ~ 480 小时
            x_train = []
            y_train = []
            for _ in range(0, _len - input_window - step + 1):
                x_train.append(input_data[_: _ + input_window])
                y_train.append(input_data[_ + input_window + step - 1: _ + input_window + step - 1 + output_window])
            x_train = np.array(_flatten(x_train))
            y_train = np.array(_flatten(y_train))
            self.x_train = x_train[:-web_split_len]
            self.y_train = y_train[:-web_split_len]
            self.x_val = x_train[-web_split_len:]
            self.y_val = y_train[-web_split_len:]

        if machine_learning:
            self.x_train, self.y_train, self.x_val, self.y_val = np.squeeze(self.x_train), \
                                                                 np.squeeze(self.y_train), \
                                                                 np.squeeze(self.x_val), \
                                                                 np.squeeze(self.y_val)
        else:
            self.x_train, self.y_train, self.x_val, self.y_val = \
                torch.from_numpy(self.x_train).to(torch.float32), \
                torch.from_numpy(self.y_train).to(torch.float32), \
                torch.from_numpy(self.x_val).to(torch.float32), \
                torch.from_numpy(self.y_val).to(torch.float32)

        if self.date_index is not None:
            self.train_index = self.date_index[0:len(self.x_train)]
            self.val_index = self.date_index[-len(self.x_val):]

        if specify_model == self.MODEL_LSTM:
            self.x_train = np.squeeze(self.x_train)
            self.y_train = np.squeeze(self.x_train)
            self.x_val = np.squeeze(self.x_val)
            self.y_val = np.squeeze(self.y_val)

        if len(self.x_train) != len(self.y_train) or len(self.x_val) != len(self.y_val):
            self.x_train, self.y_train, self.x_val, self.y_val = [], [], [], []
            print('[Error] _split... len is not correct.')

        return self.x_train, self.y_train, self.x_val, self.y_val

    def get_val_index(self):
        if self.date_index is None or self.ratio == 0.0:
            print('get_val_index... Invald.')
            return []
        return self.val_index


if __name__ == '__main__':
    dataset = DataSet(start_date='2020-01-03', end_date='2020-01-06')
