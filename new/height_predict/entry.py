import math
import pickle

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from torch import nn

from new.Util.MathUtil import MathUtil
from new.data.DataUtil import DataUtils
from new.height_predict.dataset import DataSet
from new.height_predict.eva import Eval
from new.height_predict.model.HA import HA
from new.height_predict.model.lstm import LSTM

mod_dir = './saved_model'


class PredictModel:
    MODEL_HA = 'HA'
    MODEL_SVR = 'svr'
    MODEL_KNN = 'knn'
    MODEL_DECISION_TREE = 'decision_tree'
    MODEL_RF = 'random_forest'
    MODEL_GBRT = 'GBRT'
    MODEL_LSTM = 'lstm'

    MATHS = [MODEL_HA]
    ML = [MODEL_SVR, MODEL_KNN, MODEL_DECISION_TREE, MODEL_RF, MODEL_GBRT]
    DL = [MODEL_LSTM]

    def __init__(self, station_num=1, source='../height_model/merged/stn_54511.xlsx',
                 start_date='1/1/2020', end_date='12/31/2021'):
        self.ha = HA()
        self.svr = SVR(kernel='linear')
        self.knn = KNeighborsRegressor()
        self.decision_tree = DecisionTreeRegressor()
        self.rf = RandomForestRegressor()
        self.gbrt = GradientBoostingRegressor()

        self.station_num = station_num
        self.lstm = LSTM(input_size=station_num, hidden_size=8, num_layers=1, output_size=1)
        self.entries = {self.MODEL_LSTM: self.lstm,
                        self.MODEL_SVR: self.svr,
                        self.MODEL_KNN: self.knn,
                        self.MODEL_DECISION_TREE: self.decision_tree,
                        self.MODEL_RF: self.rf,
                        self.MODEL_GBRT: self.gbrt,
                        self.MODEL_HA: self.ha}
        self.dataset = DataSet(source=source, init_feature=True, start_date=start_date, end_date=end_date)

    def retrain(self):
        # todo 生成的训练文件是否要一并清除
        self.ha = HA()
        self.svr = SVR(kernel='linear')
        self.knn = KNeighborsRegressor()
        self.decision_tree = DecisionTreeRegressor()
        self.rf = RandomForestRegressor()
        self.gbrt = GradientBoostingRegressor()
        self.lstm = LSTM(input_size=self.station_num, hidden_size=8, num_layers=1, output_size=1)

    def inner_train_maths(self, model, model_name: str, train_ratio=0.9):
        mae, rmse, mape = model.fit_predict(self.dataset, train_ratio=train_ratio)
        print('inner_train_maths... Model: {} results:'.format(model_name))
        print('mae: {:.4f}, rmse: {:.4f}, mape: {:.4f}'.format(mae, rmse, mape))
        return mae, rmse, mape

    def inner_train_ml(self, model, model_name: str, input_window=6, train_ratio=0.9, step=1, single_step=False):
        maes, rmses, mapes, predicts = [], [], [], []
        if not single_step:
            for stp in range(0, step):
                x_train, y_train, x_val, y_val = self.dataset.split(self.dataset.data, train_ratio, 1 - train_ratio,
                                                                    input_window=input_window, output_window=1,
                                                                    machine_learning=True, step=step)

                model.fit(x_train, y_train)
                # save
                f = open('{}/{}_step{}.pickle'.format(mod_dir, model_name, step + 1), 'wb')
                pickle.dump(model, f)
                f.close()
                # load
                f = open('{}/{}_step{}.pickle'.format(mod_dir, model_name, step + 1), 'rb')
                model = pickle.load(f)
                pred = model.predict(x_val)

                reals = self.dataset.inverse_transform(y_val)
                _predicts = self.dataset.inverse_transform(pred)
                mae, rmse, mape = Eval.get_matrix(_predicts, reals)
                maes.append(mae)
                rmses.append(rmse)
                mapes.append(mape)
                predicts.append(_predicts)
            # MathUtil.round(np.mean(mae), np.m)
            mae, rmse, mape = MathUtil.round(np.mean(maes), np.mean(rmses), np.mean(mapes), decimal=4)
            for _ in range(step):
                print(f'Model {model_name} result step {_ + 1}:')
                print(f'mae: {maes[_]} rmse: {rmses[_]} mape: {mapes[_]}')
        else:
            x_train, y_train, x_val, y_val = self.dataset.split(self.dataset.data, train_ratio, 1 - train_ratio,
                                                                input_window=input_window, output_window=1,
                                                                machine_learning=True, step=step)
            model.fit(x_train, y_train)
            # save
            f = open('{}/{}_step{}.pickle'.format(mod_dir, model_name, step + 1), 'wb')
            pickle.dump(model, f)
            f.close()
            # load
            f = open('{}/{}_step{}.pickle'.format(mod_dir, model_name, step + 1), 'rb')
            model = pickle.load(f)
            pred = model.predict(x_val)

            reals = self.dataset.inverse_transform(y_val)
            _predicts = self.dataset.inverse_transform(pred)
            mae, rmse, mape = Eval.get_matrix(_predicts, reals)
        if single_step:
            print('inner_train_ml... Model: {} step {} results: mae: {:.4f}, rmse: {:.4f}, mape: {:.4f}'.format(
                model_name, step, mae, rmse, mape))
        else:
            print('inner_train_ml... Model: {} avg results: mae: {:.4f}, rmse: {:.4f}, mape: {:.4f}'.format(
                model_name, step, mae, rmse, mape))
        return mae, rmse, mape, predicts

    def inner_train_dl(self, model: LSTM, model_name: str, epoch=1000,
                       input_window=6, output_window=1, train_ratio=0.9):
        loss_fn = nn.MSELoss()
        x_train, y_train, x_val, y_val = self.dataset.split(self.dataset.data, train_ratio, 1 - train_ratio,
                                                            input_window=input_window, output_window=output_window,
                                                            machine_learning=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        min_val_loss = math.inf
        cnt = 0
        model.train()
        for e in range(epoch):
            out = model(x_train)
            loss = loss_fn(out, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss < min_val_loss:
                cnt = 0
                min_val_loss = loss
                print('Better model saving. Epoch: {}. loss: {}'.format(e, loss))
                torch.save(model.state_dict(), '{}/{}.pth'.format(mod_dir, model_name))
            else:
                cnt += 1
                if cnt == 20:
                    print('>>>>>> Early stopping at epoch {}'.format(e))
                    break

        model.load_state_dict(torch.load('{}/{}.pth'.format(mod_dir, model_name),
                                         map_location=lambda storage, loc: storage))
        lstm = model.eval()
        with torch.no_grad():
            reals = []
            predicts = []
            start_idx = len(x_train)
            all_x_data = torch.cat((x_train, x_val))
            all_y_data = torch.cat((y_train, y_val))
            for _ in range(len(x_val) - 1, len(x_val)):
                out = lstm(all_x_data[:start_idx + _])
                true = all_y_data[:start_idx + _]
                reals.append(true)
                predicts.append(out)

            reals = np.concatenate(reals)
            predicts = np.concatenate(predicts)

            # 消除多余维度
            reals = np.squeeze(reals)
            predicts = np.squeeze(predicts)

            reals = self.dataset.inverse_transform(reals)
            predicts = self.dataset.inverse_transform(predicts)
            mae, rmse, mape = Eval.get_matrix(predicts, reals)
            print('inner_train_dl... Model: {} results:'.format(model_name))
            print('mae: {:.4f}, rmse: {:.4f}, mape: {:.4f}'.format(mae, rmse, mape))
            return mae, rmse, mape, predicts[:len(y_val)]

    def predict(self, select_model=MODEL_LSTM, epoch=50, input_window=6, output_window=1, with_result=False,
                train_ratio=0.9, step=1, single_step=False):
        if select_model not in self.entries.keys():
            print('predict... {} not supported'.format(select_model))
        mae = rmse = mape = -1
        predicts = None
        if select_model in self.DL:
            mae, rmse, mape, predicts = self.inner_train_dl(self.entries[select_model], select_model, epoch=epoch,
                                                            input_window=input_window, output_window=output_window,
                                                            train_ratio=train_ratio)
        elif select_model in self.ML:
            mae, rmse, mape, predicts = self.inner_train_ml(self.entries[select_model], select_model,
                                                            input_window=input_window, train_ratio=train_ratio,
                                                            step=step, single_step=single_step)
        elif select_model in self.MATHS:
            mae, rmse, mape = self.inner_train_maths(self.entries[select_model], select_model, train_ratio=train_ratio)
        else:
            print('predict... Model unclassified.')
        if with_result:
            return mae, rmse, mape, predicts
        else:
            return mae, rmse, mape

    @staticmethod
    def predict_and_eval_all_models(source='../height_model/merged/stn_59758.xlsx', output_root='./results/',
                                    train_ratio=0.9,
                                    input_window=6, epoch=1000, step=2, single_step=True):
        """
        为每一个特征都，使用所有模型计算评价指标记录到 txt，纵坐标为模型
        """
        output_window = 1
        features = [DataSet.FEATURE_NPS, DataSet.FEATURE_BABIN, DataSet.FEATURE_LIULI, DataSet.FEATURE_PJ]
        models = PredictModel.ML  # todo DL
        wb, ws, output_name = DataUtils.excel_writer_prepare(header=['model', 'mae', 'rmse', 'mape'],
                                                             output_name=output_root + 'all_model_eval')
        pm = PredictModel(source=source)
        for feature in features:
            pm.dataset.change_feature(feature)
            for model in models:
                for stp in range(1, step + 1):
                    mae, rmse, mape = pm.predict(select_model=model, epoch=epoch,
                                                 input_window=input_window, output_window=output_window,
                                                 with_result=False, train_ratio=train_ratio, step=stp,
                                                 single_step=single_step)
                    ws.append([f'{DataSet.FEATURE_NAME_MAP[feature]}_{model}_step_{stp}', mae, rmse, mape])
        wb.save(filename=output_name)
        print('predict_and_eval_all_models... Fin.')

    @staticmethod
    def predict_and_record_all_models(source='../height_model/merged/stn_59758.xlsx', output_root='./results/',
                                      log_dir='./results/stn_59758.txt', train_ratio=0.9,
                                      input_window=6, epoch=1000):
        """
        为每一个特征都，使用所有模型计算评价指标记录到 txt，然后生成预测值，写入 excel，一个特征一个 excel，纵坐标为模型
        [只适用于单步预测]
        """
        output_window = 1
        pm = PredictModel(source=source)
        features = [DataSet.FEATURE_NPS, DataSet.FEATURE_BABIN, DataSet.FEATURE_LIULI, DataSet.FEATURE_PJ]
        models = PredictModel.ML + PredictModel.DL
        with open(log_dir, 'a+') as f:
            f.truncate(0)
            for feature in features:
                f.write('---------Model [{}]---------\n'.format(DataSet.FEATURE_NAME_MAP[feature]))
                pm.dataset.change_feature(feature)
                wb, ws, output_name = DataUtils.excel_writer_prepare(header=['date', 'real'] + models,
                                                                     output_name=output_root + DataSet.FEATURE_NAME_MAP[
                                                                         feature])
                _1, _2, _3, true = pm.dataset.split(pm.dataset.data, train_ratio, 1 - train_ratio,
                                                    input_window=input_window, output_window=output_window,
                                                    machine_learning=True)
                # 时间
                date_index = np.array(pm.dataset.get_val_index().tolist())
                # 真实值
                true = pm.dataset.inverse_transform(true)
                results_list = [date_index, true[:, 0]]
                for model in models:
                    mae, rmse, mape, results = pm.predict(select_model=model, epoch=epoch,
                                                          input_window=input_window, output_window=output_window,
                                                          with_result=True, train_ratio=train_ratio)
                    results_list.append(results[:, 0])
                    f.write('{}>>>>>\n'.format(model))
                    f.write(f'MAE: {mae} , RMSE: {rmse} , MAPE : {mape}\n')

                results_list = np.array(results_list).swapaxes(0, 1)
                for _ in range(0, len(results_list)):
                    ws.append(results_list[_].tolist())
                wb.save(filename=output_name)


if __name__ == '__main__':
    pm = PredictModel()
    # pm.predict(select_model=PredictModel.MODEL_LSTM, epoch=1000)
    # pm.predict(select_model=PredictModel.MODEL_SVR)
    # pm.predict(select_model=PredictModel.MODEL_KNN)
    # pm.predict(select_model=PredictModel.MODEL_DECISION_TREE)
    # pm.predict(select_model=PredictModel.MODEL_RF)
    # pm.predict(select_model=PredictModel.MODEL_GBRT, step=2)
    # pm.predict(select_model=PredictModel.MODEL_HA)

    # PredictModel.predict_and_record_all_models(source='../height_model/merged/stn_59758.xlsx', epoch=1000)
    PredictModel.predict_and_eval_all_models(source='../height_model/merged/stn_59758.xlsx')
