import numpy as np

from new.height_predict.dataset import DataSet
from new.height_predict.eva import Eval


class HA:

    @staticmethod
    def fit_predict(data: DataSet, train_ratio=0.9, window=3):
        """
        :param data:
        :param train_ratio: 用来厘定测试范围
        :param window: 拿前 window 天的平均值作为预测值
        :return:
        """
        preds = []
        reals = []
        _data = data.self_inverse_transform()  # (B, F)

        n = len(_data)
        train_len = int(n * train_ratio)
        if train_len - window < 0:
            print('[Error] HA... fit_predict cannot perform. Training set is too short.')
            return -1, -1, -1
        for _ in range(train_len, n):
            real = _data[_]
            # 往前找 window 个节点
            forward = _data[train_len-window:train_len,:]
            pred = np.mean(forward, axis=0)
            preds.append(pred)
            reals.append(real)
        preds = np.concatenate(preds)
        reals = np.concatenate(reals)
        return Eval.get_matrix(preds, reals)
