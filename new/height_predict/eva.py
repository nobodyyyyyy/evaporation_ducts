import numpy as np

from Util.MathUtil import MathUtil


class Eval:

    @staticmethod
    def RMSE(y_pred: np.array, y_true: np.array):
        return np.sqrt(np.mean(np.square(y_pred - y_true)))

    @staticmethod
    def MAE(y_pred: np.array, y_true: np.array):
        return np.mean(np.abs(y_pred - y_true))

    @staticmethod
    def MAPE(y_pred: np.array, y_true: np.array, epsilon=1e-0):  # zero division
        return np.mean(np.abs(y_pred - y_true) / (y_true + epsilon))

    @staticmethod
    def PCC(y_pred: np.array, y_true: np.array):
        return np.corrcoef(y_pred.flatten(), y_true.flatten())[0, 1]

    @staticmethod
    def get_matrix(y_pred: np.array, y_true: np.array):
        mae, rmse, mape = Eval.MAE(y_pred, y_true), Eval.RMSE(y_pred, y_true), Eval.MAPE(y_pred, y_true)
        return MathUtil.round(mae, rmse, mape, decimal=4)
