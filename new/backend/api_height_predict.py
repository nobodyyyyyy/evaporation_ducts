import re

import pandas as pd
from flask import request, Blueprint
import json
import numpy as np

from Util.TimeUtil import TimeUtil
from backend.SupportedSingletons import SupportedSingletons
from height_predict.entry import PredictModel

height_predict_api = Blueprint('heightPredict', __name__)

# 单例列表
modules = SupportedSingletons()

prefix = '/predict'

API_PREDICT = prefix + '/predict'

from config import DEBUG
if not DEBUG:
    _prefix = '../../height_model'
else:
    _prefix = './height_model'

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
    output_len = data['output_len']
    predict_len = int(re.findall(r".*\((.*)天",output_len)[0])
    # print(predict_len)
    predict_time = predict_len * 24 * 60 * 60
    # 拿文件名
    file_addr = modules.dataUtil.get_duct_info_file_address(station_id=_id, prefix=_prefix)
    date_f = str(TimeUtil.timestamp_to_datetime(date_from))
    train_date_t = str(TimeUtil.timestamp_to_datetime(date_to))
    date_t = str(TimeUtil.timestamp_to_datetime(date_to + predict_time))  # 预测未来48小时
    model = PredictModel(station_num=1, source=file_addr, start_date=date_f, end_date=date_t, feature_name=height_model)
    mae, rmse, mape = model.predict(select_model=predict_model, epoch=epoch,
                                    input_window=input_window, output_window=1,
                                    with_result=False, step=1, single_step=False, pso_optimize=pso, web_split=True, web_split_len=predict_len)
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
            'mape': mape,
            'output_len': output_len
        }
    })
