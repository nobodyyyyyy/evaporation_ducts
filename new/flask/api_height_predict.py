import re

import pandas as pd
from flask import request, Blueprint
import json
import numpy as np

from new.Util.TimeUtil import TimeUtil
from new.flask.SupportedSingletons import SupportedSingletons
from new.height_predict.entry import PredictModel

height_predict_api = Blueprint('heightPredict', __name__)

# 单例列表
modules = SupportedSingletons()

prefix = '/predict'

API_PREDICT = prefix + '/predict'


@height_predict_api.route(API_PREDICT, methods=['POST', 'GET'])
def init_entry():
    data = request.get_json()
    date_from = data['date'][0] / 1000
    date_to = data['date'][1] / 1000
    _id = data['id']
    pso = data['pso']
    predict_model = data['predict_model']
    height_model = data['height_model']
    epoch = data['epoch']
    input_window = data['window']

    # 拿文件名
    file_addr = modules.dataUtil.get_duct_info_file_address(station_id=_id, prefix='../height_model')
    date_f = str(TimeUtil.timestamp_to_datetime(date_from))
    train_date_t = str(TimeUtil.timestamp_to_datetime(date_to))
    date_t = str(TimeUtil.timestamp_to_datetime(date_to + 48 * 60 * 60))  # 预测未来48小时
    model = PredictModel(station_num=1, source=file_addr, start_date=date_f, end_date=date_t, feature_name=height_model)
    mae, rmse, mape = model.predict(select_model=predict_model, epoch=epoch,
                                    input_window=input_window, output_window=1,
                                    with_result=False, step=1, single_step=False, pso_optimize=pso, web_split=True)
    if height_model == 'babin':
        height_model = 'byc'
    if height_model == 'liuli':
        height_model = 'mgb'
    if predict_model in ['LSTM(RNN)', 'GRU(RNN)']:
        pso = '不支持'

    return json.dumps({
        'code': 0,
        'msg': '训练完成，结果表格已更新',
        'table_entry': {
            'date': TimeUtil.current_time_str(),
            'station': _id,
            'height_model': height_model,
            'predict_model': predict_model,
            'pso': str(pso),
            'range': f'{date_f}~{train_date_t}',
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }
    })
