from math import sqrt

from sklearn.metrics import mean_squared_error, mean_absolute_error


class EvalUtil:

    @staticmethod
    def eval(test, pred, model=''):
        mse = mean_squared_error(test, pred)
        mae = mean_absolute_error(test, pred)
        rmse = sqrt(mse)
        print('----------{}_eval----------'.format(model))
        print('mse: {}'.format(round(mse, 3)))
        print('mae: {}'.format(round(mae, 3)))
        print('rmse: {}'.format(round(rmse, 3)))


if __name__ == '__main__':
    pass