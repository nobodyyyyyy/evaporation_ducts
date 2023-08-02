import math
import pickle

import numpy as np
import torch
import torch.utils.data as Data
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

    DATA_SPLIT_CONDITION = {MODEL_LSTM: DataSet.MODEL_LSTM}

    def __init__(self, station_num=1, source='../height_model/merged/stn_54511.xlsx',
                 start_date='1/1/2020', end_date='12/31/2021'):
        self.ha = HA()
        self.svr = SVR(kernel='linear')
        self.knn = KNeighborsRegressor()
        self.decision_tree = DecisionTreeRegressor()
        self.rf = RandomForestRegressor()
        self.gbrt = GradientBoostingRegressor()

        self.station_num = station_num
        self.lstm = LSTM(input_size=6, hidden_size=8, num_layers=1, output_size=1)
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
            predicts = self.dataset.inverse_transform(pred)
            mae, rmse, mape = Eval.get_matrix(predicts, reals)
        if single_step:
            print('inner_train_ml... Model: {} step {} results: mae: {:.4f}, rmse: {:.4f}, mape: {:.4f}'.format(
                model_name, step, mae, rmse, mape))
        else:
            print('inner_train_ml... Model: {} avg results: mae: {:.4f}, rmse: {:.4f}, mape: {:.4f}'.format(
                model_name, step, mae, rmse, mape))
        return mae, rmse, mape, predicts

    def inner_train_lstm(self, model: LSTM, model_name: str, epoch=1000,
                         input_window=6, output_window=1, train_ratio=0.9):
        split_condition = self.DATA_SPLIT_CONDITION[self.MODEL_LSTM]
        x_train, y_train, x_val, y_val = self.dataset.split(self.dataset.data, train_ratio, 1 - train_ratio,
                                                            input_window=input_window, output_window=output_window,
                                                            machine_learning=False, specify_model=split_condition)

        train_data = Data.TensorDataset(x_train, y_train)
        test_data = Data.TensorDataset(x_val, y_val)
        train_loader = Data.DataLoader(dataset=train_data, batch_size=4, shuffle=False)
        test_loader = Data.DataLoader(dataset=test_data, batch_size=4, shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_func = nn.MSELoss()
        min_val_loss = math.inf
        train_loss_all = []
        model.train()
        for e in range(epoch):
            train_loss = 0
            train_num = 0
            for step, (b_x, b_y) in enumerate(train_loader):
                output = model(b_x)
                loss = loss_func(output, b_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * b_x.size(0)
                train_num += b_x.size(0)

            # print(f'Epoch{epoch + 1}/{epoch}: Loss:{train_loss / train_num}')
            los = train_loss / train_num
            print('Epoch: {}. loss: {}'.format(e, round(los, 4)))
            # train_loss_all.append(los)
            if los < min_val_loss:
                cnt = 0
                min_val_loss = los
                print('Better model saving. Epoch: {}. loss: {}'.format(e, los))
                torch.save(model.state_dict(), '{}/{}.pth'.format(mod_dir, model_name))
            else:
                cnt += 1
                if cnt == 100:
                    print('>>>>>> Early stopping at epoch {}'.format(e))
                    break

        model.load_state_dict(torch.load('{}/{}.pth'.format(mod_dir, model_name),
                                         map_location=lambda storage, loc: storage))
        model = model.eval()
        with torch.no_grad():
            reals = []
            predicts = []
            for step, (b_x, b_y) in enumerate(test_loader):
                output = model(b_x)
                reals.append(b_y)
                output = output.squeeze(-1)
                predicts.append(output)

            _reals = torch.stack(reals[:-1]).reshape(-1)
            _predicts = torch.stack(predicts[:-1]).reshape(-1)

            _reals = torch.concat([_reals, reals[-1]])
            _predicts = torch.concat([_predicts, predicts[-1]])

            _reals = self.dataset.inverse_transform(_reals)
            _predicts = self.dataset.inverse_transform(_predicts)

            mae, rmse, mape = Eval.get_matrix(_predicts, _reals)
            print('inner_train_dl... Model: {} results:'.format(model_name))
            print('mae: {:.4f}, rmse: {:.4f}, mape: {:.4f}'.format(mae, rmse, mape))
            return mae, rmse, mape, _predicts[:len(y_val)]

    def predict(self, select_model=MODEL_LSTM, epoch=50, input_window=6, output_window=1, with_result=False,
                train_ratio=0.9, step=1, single_step=False):
        if select_model not in self.entries.keys():
            print('predict... {} not supported'.format(select_model))
        mae = rmse = mape = -1
        predicts = None
        if select_model == self.MODEL_LSTM:
            mae, rmse, mape, predicts = self.inner_train_lstm(self.entries[select_model], select_model, epoch=epoch,
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
        # features = [DataSet.FEATURE_NPS, DataSet.FEATURE_BABIN, DataSet.FEATURE_LIULI, DataSet.FEATURE_PJ]
        features = [DataSet.FEATURE_NPS]
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
                                                          with_result=True, train_ratio=train_ratio,
                                                          single_step=True)
                    results_list.append(results[:, 0])
                    f.write('{}>>>>>\n'.format(model))
                    f.write(f'MAE: {mae} , RMSE: {rmse} , MAPE : {mape}\n')

                results_list = np.array(results_list).swapaxes(0, 1)
                for _ in range(0, len(results_list)):
                    ws.append(results_list[_].tolist())
                wb.save(filename=output_name)


if __name__ == '__main__':
    pm = PredictModel()
    # pm.predict(select_model=PredictModel.MODEL_LSTM, epoch=100)
    # pm.predict(select_model=PredictModel.MODEL_SVR)
    # pm.predict(select_model=PredictModel.MODEL_KNN)
    # pm.predict(select_model=PredictModel.MODEL_DECISION_TREE)
    # pm.predict(select_model=PredictModel.MODEL_RF)
    # pm.predict(select_model=PredictModel.MODEL_GBRT, step=2)
    # pm.predict(select_model=PredictModel.MODEL_HA)

    PredictModel.predict_and_record_all_models(source='../height_model/merged/stn_59758.xlsx', epoch=1000)
    # PredictModel.predict_and_eval_all_models(source='../height_model/merged/stn_59758.xlsx')
