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
from torch.utils.data import TensorDataset
from tqdm import tqdm

from Util.MathUtil import MathUtil
from data.DataUtil import DataUtils
from height_predict.dataset import DataSet
from height_predict.eva import Eval
from height_predict.model.HA import HA
from height_predict.model.lstm import RNN, LSTMnetwork, GRU
from height_predict.pso import PSO

mod_dir = './saved_model'


class PredictModel:
    SEED = 1

    MODEL_HA = 'HA'
    MODEL_SVR = 'SVR'
    MODEL_KNN = 'KNN'
    MODEL_DECISION_TREE = 'DT'
    MODEL_RF = 'RF'
    MODEL_GBRT = 'GBRT'
    MODEL_LSTM = 'LSTM(RNN)'
    MODEL_RNN = 'rnn(THERE_IS_NO_SUCH_METHOD)'
    MODEL_GRU = 'GRU(RNN)'

    FEATURE_NPS = 7
    FEATURE_BABIN = 8
    FEATURE_LIULI = 9
    FEATURE_PJ = 10
    FEATURE_WEIZHESHELV = 11

    FEATURE_COL_MAP = {
        'nps': FEATURE_NPS,
        'babin': FEATURE_BABIN,
        'liuli': FEATURE_LIULI,
        'pj': FEATURE_PJ,
        'wei': FEATURE_WEIZHESHELV
    }

    MATHS = [MODEL_HA]
    ML = [MODEL_SVR, MODEL_KNN, MODEL_DECISION_TREE, MODEL_RF, MODEL_GBRT]
    DL = [MODEL_LSTM, MODEL_RNN]

    DATA_SPLIT_CONDITION = {MODEL_LSTM: DataSet.MODEL_LSTM, MODEL_RNN: DataSet.MODEL_RNN}

    def __init__(self, station_num=1, source='../height_model/merged/stn_59758.xlsx',
                 start_date='2020-01-01', end_date='2021-12-31', feature_name='nps'):
        self.station_num = station_num
        self.entries = {self.MODEL_LSTM: LSTMnetwork(input_size=1, hidden_size=100, output_size=1),
                        self.MODEL_RNN: RNN(input_size=1, hidden_size=8, num_layers=1, output_size=1),
                        self.MODEL_GRU: GRU(feature_size=1, hidden_size=8, output_size=1),
                        self.MODEL_SVR: SVR(kernel='linear', ),
                        self.MODEL_KNN: KNeighborsRegressor(),
                        self.MODEL_DECISION_TREE: DecisionTreeRegressor(random_state=PredictModel.SEED),
                        self.MODEL_RF: RandomForestRegressor(random_state=PredictModel.SEED),
                        self.MODEL_GBRT: GradientBoostingRegressor(random_state=PredictModel.SEED),
                        self.MODEL_HA: HA()}
        self.dataset = DataSet(source=source, init_feature=True,
                               start_date=start_date, end_date=end_date, col=self.FEATURE_COL_MAP[feature_name])

        try:
            torch.manual_seed(PredictModel.SEED)
            torch.cuda.manual_seed_all(PredictModel.SEED)
            np.random.seed(PredictModel.SEED)
            torch.backends.cudnn.deterministic = True
        except Exception as e:
            print('entry init... Err: {}'.format(str(e)))
            pass

    def retrain_all(self):
        # todo 生成的训练文件是否要一并清除
        self.entries = {self.MODEL_LSTM: LSTMnetwork(input_size=1, hidden_size=100, output_size=1),
                        self.MODEL_RNN: RNN(input_size=1, hidden_size=8, num_layers=1, output_size=1),
                        self.MODEL_GRU: GRU(feature_size=1, hidden_size=8, output_size=1),
                        self.MODEL_SVR: SVR(kernel='linear'),
                        self.MODEL_KNN: KNeighborsRegressor(),
                        self.MODEL_DECISION_TREE: DecisionTreeRegressor(random_state=PredictModel.SEED),
                        self.MODEL_RF: RandomForestRegressor(random_state=PredictModel.SEED),
                        self.MODEL_GBRT: GradientBoostingRegressor(random_state=PredictModel.SEED),
                        self.MODEL_HA: HA()}

    def retrain(self, model_name):
        if model_name == self.MODEL_SVR:
            self.entries[self.MODEL_SVR] = SVR(kernel='linear')
        elif model_name == self.MODEL_KNN:
            self.entries[self.MODEL_KNN] = KNeighborsRegressor()
        elif model_name == self.MODEL_DECISION_TREE:
            self.entries[self.MODEL_DECISION_TREE] = DecisionTreeRegressor(random_state=1)
        elif model_name == self.MODEL_RF:
            self.entries[self.MODEL_RF] = RandomForestRegressor(random_state=1)
        elif model_name == self.MODEL_GBRT:
            self.entries[self.MODEL_GBRT] = GradientBoostingRegressor()
        elif model_name == self.MODEL_HA:
            self.entries[self.MODEL_HA] = HA()
        elif model_name == self.MODEL_LSTM:
            self.entries[self.MODEL_LSTM] = LSTMnetwork(input_size=1, hidden_size=8, output_size=1)
        elif model_name == self.MODEL_RNN:
            self.entries[self.MODEL_RNN] = RNN(input_size=1, hidden_size=8, num_layers=1, output_size=1)
        elif model_name == self.MODEL_GRU:
            self.entries[self.MODEL_GRU] = GRU(feature_size=1, hidden_size=8, output_size=1)

    def inner_train_maths(self, model, model_name: str, train_ratio=0.9, web_split=False, web_split_len=2):
        mae, rmse, mape = model.fit_predict(self.dataset, train_ratio=train_ratio)
        print('inner_train_maths... Model: {} results:'.format(model_name))
        print('mae: {:.4f}, rmse: {:.4f}, mape: {:.4f}'.format(mae, rmse, mape))
        return mae, rmse, mape

    def inner_train_ml(self, model, model_name: str, input_window=6, train_ratio=0.9, step=1, single_step=False,
                       pso_optimize=False, web_split=False, web_split_len=2):
        maes, rmses, mapes, predicts = [], [], [], []

        if single_step:
            steps = [step]
        else:
            steps = list(range(1, step + 1))

        for stp in steps:
            x_train, y_train, x_val, y_val = self.dataset.split(self.dataset.data, train_ratio, 1 - train_ratio,
                                                                input_window=input_window, output_window=1,
                                                                machine_learning=True, step=stp, web_split=web_split,
                                                                web_split_len=web_split_len)

            if pso_optimize:
                pso = PSO(model_name=model_name, seed=self.SEED)
                model = pso.run(x_train, y_train, x_val, y_val)
                self.entries[model_name] = model

            model.fit(x_train, y_train)
            # # save
            # f = open('{}/{}_step{}.pickle'.format(mod_dir, model_name, step), 'wb')
            # pickle.dump(model, f)
            # f.close()
            # # load
            # f = open('{}/{}_step{}.pickle'.format(mod_dir, model_name, step), 'rb')
            # model = pickle.load(f)
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

        if single_step:
            print('inner_train_ml... Model: {} step {} results: mae: {}, rmse: {}, mape: {}'.format(
                model_name, step, mae, rmse, mape))
        else:
            print('inner_train_ml... Model: {} avg results: mae: {}, rmse: {}, mape: {}'.format(
                model_name, mae, rmse, mape))
        return mae, rmse, mape, predicts

    @DeprecationWarning
    def old_inner_train_lstm(self, model=None, model_name='', epoch=1000,
                             input_window=6, output_window=1, train_ratio=0.9, web_split=False):
        split_condition = self.DATA_SPLIT_CONDITION[self.MODEL_LSTM]
        x_train, y_train, x_val, y_val = self.dataset.split(self.dataset.data, train_ratio, 1 - train_ratio,
                                                            input_window=input_window, output_window=output_window,
                                                            machine_learning=False, specify_model=split_condition,
                                                            web_split=web_split)

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

    def inner_train_lstm(self, model=None, model_name='', epoch=1000,
                         input_window=6, output_window=1, train_ratio=0.9, web_split=False, web_split_len=2):
        model.train()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        x_train, y_train, x_val, y_val = self.dataset.split(self.dataset.data, train_ratio, 1 - train_ratio,
                                                            input_window=input_window, output_window=output_window,
                                                            machine_learning=False, web_split=web_split, web_split_len=web_split_len)
        y_train = y_train.squeeze(dim=-1)
        y_val = y_val.squeeze(dim=-1)
        train_len = x_train.shape[0]
        val_len = x_val.shape[0]
        for e in range(epoch):
            e_loss = []
            for _ in range(train_len):
                optimizer.zero_grad()
                model.hidden = (torch.zeros(1, 1, model.hidden_size),
                                torch.zeros(1, 1, model.hidden_size))

                y_pred = model(x_train[_])

                loss = criterion(y_pred, y_train[_])
                e_loss.append(loss.item())
                loss.backward()
                optimizer.step()
            mean_loss = np.mean(e_loss)
            print(f'inner_train_lstm... Model: LSTM Epoch: {e + 1} Loss: {mean_loss:10.4f}')

        model.eval()
        preds = []
        for _ in range(val_len):
            with torch.no_grad():
                model.hidden = (torch.zeros(1, 1, model.hidden_size), torch.zeros(1, 1, model.hidden_size))
                preds.append(model(x_val[_]).item())
        _predicts = self.dataset.inverse_transform(np.array(preds))
        _reals = self.dataset.inverse_transform(y_val)

        mae, rmse, mape = Eval.get_matrix(_predicts, _reals)
        print('inner_train_lstm... Model: {} results:'.format(model_name))
        print('mae: {:.4f}, rmse: {:.4f}, mape: {:.4f}'.format(mae, rmse, mape))
        return mae, rmse, mape, _predicts

    def inner_train_gru(self, model=None, model_name='', epoch=1000,
                        input_window=6, output_window=1, train_ratio=0.9, batch_size=24, web_split=False, web_split_len=2):

        model.train()
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        x_train, y_train, x_val, y_val = self.dataset.split(self.dataset.data, train_ratio, 1 - train_ratio,
                                                            input_window=input_window, output_window=output_window,
                                                            machine_learning=False, web_split=web_split, web_split_len=web_split_len)
        # train_data = TensorDataset(x_train, y_train)
        # test_data = TensorDataset(x_val, y_val)
        # train_loader = torch.utils.data.DataLoader(train_data, batch_size, False)
        # test_loader = torch.utils.data.DataLoader(test_data, batch_size, False)
        for e in range(epoch):
            model.train()
            loss_arr = []
            # train_bar = tqdm(train_loader)  # 形成进度条
            for _ in range(x_train.shape[0]):
                x = x_train[_]
                y = y_train[_]
                optimizer.zero_grad()
                y_train_pred = model(x)
                y = y.reshape(-1, 1)
                loss = loss_function(y_train_pred, y)
                loss.backward()
                optimizer.step()

                loss_arr.append(loss.item())
            print(f'inner_train_gru... Model: GRU Epoch: {e + 1} Loss: {np.mean(loss_arr):10.4f}')
            # train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(e + 1, epoch, np.mean(loss_arr))

        # 模型验证
        y_val = y_val.squeeze(dim=-1)
        model.eval()
        preds = []
        with torch.no_grad():
            for _ in range(x_val.shape[0]):
                x = x_val[_]
                y = y_val[_]
                y_test_pred = model(x)
                preds.append(y_test_pred)
        _predicts = self.dataset.inverse_transform(np.array(preds))
        _reals = self.dataset.inverse_transform(y_val)

        mae, rmse, mape = Eval.get_matrix(_predicts, _reals)
        print('inner_train_gru... Model: {} results:'.format(model_name))
        print('mae: {:.4f}, rmse: {:.4f}, mape: {:.4f}'.format(mae, rmse, mape))
        return mae, rmse, mape, _predicts

    def predict(self, select_model=MODEL_LSTM, epoch=1000, input_window=6, output_window=1, with_result=False,
                train_ratio=0.9, step=1, single_step=False, pso_optimize=False, web_split=False, web_split_len=2):
        if select_model not in self.entries.keys():
            print('predict... {} not supported'.format(select_model))
        mae = rmse = mape = -1
        predicts = None

        # todo 要加一个不重新训练的逻辑
        self.retrain(model_name=select_model)

        if select_model == self.MODEL_LSTM:
            mae, rmse, mape, predicts = self.inner_train_lstm(self.entries[select_model], select_model, epoch=epoch,
                                                              input_window=input_window,
                                                              output_window=output_window,
                                                              train_ratio=train_ratio,
                                                              web_split=web_split,
                                                              web_split_len=web_split_len)
        elif select_model == self.MODEL_RNN:
            print('not supported')
            pass
        elif select_model == self.MODEL_GRU:
            mae, rmse, mape, predicts = self.inner_train_gru(self.entries[select_model], select_model, epoch=epoch,
                                                             input_window=input_window,
                                                             output_window=output_window,
                                                             train_ratio=train_ratio,
                                                             web_split=web_split,
                                                             web_split_len=web_split_len)
        elif select_model in self.ML:
            mae, rmse, mape, predicts = self.inner_train_ml(self.entries[select_model], select_model,
                                                            input_window=input_window, train_ratio=train_ratio,
                                                            step=step, single_step=single_step,
                                                            pso_optimize=pso_optimize,
                                                            web_split=web_split,
                                                            web_split_len=web_split_len)
        elif select_model in self.MATHS:
            mae, rmse, mape = self.inner_train_maths(self.entries[select_model], select_model, train_ratio=train_ratio,
                                                     web_split=web_split,
                                                     web_split_len=web_split_len)
        else:
            print('predict... Model unclassified.')
        if with_result:
            return mae, rmse, mape, predicts
        else:
            return mae, rmse, mape

    @staticmethod
    def predict_and_eval_all_models(source='../height_model/merged/stn_59758.xlsx', output_root='./results/',
                                    output_name='all_model_eval',
                                    train_ratio=0.9,
                                    input_window=6, epoch=1000, step=2, single_step=True, pso=False):
        """
        为每一个特征都，使用所有模型计算评价指标记录到 txt，纵坐标为模型
        """
        output_window = 1
        features = [DataSet.FEATURE_NPS, DataSet.FEATURE_BABIN, DataSet.FEATURE_LIULI, DataSet.FEATURE_PJ]
        # features = [DataSet.FEATURE_NPS]
        # models = PredictModel.ML  # todo DL
        models = [PredictModel.MODEL_LSTM]
        wb, ws, output_name = DataUtils.excel_writer_prepare(header=['model', 'mae', 'rmse', 'mape'],
                                                             output_name=output_root + output_name)
        pm = PredictModel(source=source)
        for feature in features:
            pm.dataset.change_feature(feature)
            for model in models:
                for stp in range(1, step + 1):
                    mae, rmse, mape = pm.predict(select_model=model, epoch=epoch,
                                                 input_window=input_window, output_window=output_window,
                                                 with_result=False, train_ratio=train_ratio, step=stp,
                                                 single_step=single_step, pso_optimize=pso)
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
        # models = PredictModel.ML + PredictModel.DL
        models = PredictModel.ML + [PredictModel.MODEL_RNN]
        # models = [PredictModel.MODEL_LSTM]
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
                    if model in PredictModel.ML:
                        results_list.append(results[0][:, 0])
                    else:
                        results_list.append(results[:, 0])
                    f.write('{}>>>>>\n'.format(model))
                    f.write(f'MAE: {mae} , RMSE: {rmse} , MAPE : {mape}\n')

                results_list = np.array(results_list).swapaxes(0, 1)
                for _ in range(0, len(results_list)):
                    ws.append(results_list[_].tolist())
                wb.save(filename=output_name)


if __name__ == '__main__':
    pm = PredictModel(start_date='2020-01-03', end_date='2020-03-06')
    pm.predict(select_model=PredictModel.MODEL_KNN, epoch=20, web_split=True)
    # pm.predict(select_model=PredictModel.MODEL_SVR)
    # pm.predict(select_model=PredictModel.MODEL_KNN)
    # pm.predict(select_model=PredictModel.MODEL_DECISION_TREE)
    # pm.predict(select_model=PredictModel.MODEL_RF, single_step=False, pso_optimize=True)
    # pm.predict(select_model=PredictModel.MODEL_GBRT, step=1, pso_optimize=True)
    # pm.predict(select_model=PredictModel.MODEL_HA)
    # PredictModel.predict_and_record_all_models(source='../height_model/merged/stn_59758.xlsx', epoch=30)

    # PredictModel.predict_and_eval_all_models(source='../height_model/merged/stn_59758.xlsx', output_name='lstm',
    #                                          step=2, epoch=1000)
