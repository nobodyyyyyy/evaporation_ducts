import json

import flask
import numpy as np
import pandas as pd
from flask import request, Blueprint

from new.Util.TimeUtil import TimeUtil
from new.flask.SupportedSingletons import SupportedSingletons
from new.height_predict.dataset import DataSet


radar_api = Blueprint('radar', __name__)

# 高度模型对应的列
FEATURE_NPS = DataSet.FEATURE_NPS
FEATURE_BABIN = DataSet.FEATURE_BABIN
FEATURE_LIULI = DataSet.FEATURE_LIULI
FEATURE_PJ = DataSet.FEATURE_PJ
FEATURE_WEIZHESHELV = DataSet.FEATURE_WEIZHESHELV

FEATURE_COL_MAP = {
    'nps': 'nps',
    'babin': 'babin',
    'liuli': 'liuli2.0',
    'pj': 'pj'
}

# 单例列表
modules = SupportedSingletons()

prefix = '/radar'

API_GET_LOSS = prefix + '/loss_cal'
API_GET_P_DETECT = prefix + '/p_detect_cal'


@radar_api.route(API_GET_LOSS, methods=['POST', 'GET'])
def radar_cal_loss():
    data = request.get_json()
    hgt = data['hgt']
    freq = data['freq']
    self_define = data['self_define']
    eva_hgt = data['duct_height']
    if not self_define:
        eva_hgt = -1
        date = data['date'] / 1000
        _id = data['id']
        height_model = data['height_model']

        # 拿文件名
        file_addr = modules.dataUtil.get_duct_info_file_address(station_id=_id, prefix='../height_model')
        raw = pd.read_excel(file_addr)
        for _, row in raw.iterrows():
            ts = TimeUtil.str_to_timestamp(row['日期'])
            if ts >= date:
                eva_hgt = row[FEATURE_COL_MAP[height_model]]
                break
        if eva_hgt == -1:
            return json.dumps({
                'code': 0,
                'msg': '所选站点日期没有计算好的波导高度，请选择其他日期'
            })

    # 拿传输损耗矩阵
    _min = 10000000
    _max = 0
    loss = modules.radarCal.simple_get_l_single(eva_hgt, freq, hgt)
    loss = loss.tolist()
    # 看看是要拿传输损耗，还是拿盲区
    if data['search_type'] == 'loss':
        dis_arr = list(range(1, len(loss) + 1))
        hgt_arr = list(range(1, len(loss[0]) + 1))
        ret, _max, _min = modules.dataUtil.gen_data_response_4_heatmap(loss)
        if self_define:
            title = f'波导高度 {eva_hgt} m， 天线高度 {hgt} m，雷达频率 {freq} MHz'
        else:
            title = f'{_id}, {str(TimeUtil.timestamp_to_datetime(date))}, 波导高度 {eva_hgt} m， 天线高度 {hgt} m，雷达频率 {freq} MHz'
        return json.dumps({
            'code': 0,
            'msg': f'波导损耗计算完毕。波导高度 {eva_hgt} m， 天线高度 {hgt} m，雷达频率 {freq} MHz。',
            'eva_hgt': eva_hgt,
            'data': ret,
            # 'x': dis_arr,
            # 'y': hgt_arr,
            'y': dis_arr,
            'x': hgt_arr,
            'max_value': _max,
            'min_value': _min,
            'title': title
        })
    else:
        # 拿盲区
        radar_param = data['detection_param']
        pt = radar_param['pt']
        g = radar_param['G']
        d0 = radar_param['D0']
        bn = radar_param['Bn']
        ls = radar_param['Ls']
        f0 = radar_param['F0']
        sigma = radar_param['sigma']
        ts = modules.radarCal.get_Ts(freq, pt, g, d0, bn, ls, f0, sigma)
        blind = np.zeros((100, 20))
        blind = blind.tolist()
        dis_arr = list(range(1, 100))
        hgt_arr = list(range(1, 20))
        for i in range(100):
            for j in range(20):
                if not modules.radarCal.is_detected(loss[i][j], ts):
                    blind[i][j] = 1
        ret, _max, _min = modules.dataUtil.gen_data_response_4_heatmap(blind)
        if self_define:
            title = f'盲区检测 波导高度 {eva_hgt} m， 天线高度 {hgt} m，雷达频率 {freq} MHz'
        else:
            title = f'盲区检测 {_id}, {str(TimeUtil.timestamp_to_datetime(date))}, 波导高度 {eva_hgt} m， 天线高度 {hgt} m，雷达频率 {freq} MHz'
        return json.dumps({
            'code': 0,
            'msg': f'盲区计算完毕。波导高度 {eva_hgt} m， 天线高度 {hgt} m，雷达频率 {freq} MHz，雷达峰值功率 {pt} KW，天线增益 {g} dB，'
                   f'最小信噪比 {d0} dB，接收机带宽 {bn} MHz， 系统综合损耗 {ls} dB，接收机噪声系数 {f0} dB， 目标散射截面 {sigma} m^2',
            'eva_hgt': eva_hgt,
            'data': ret,
            # 'x': dis_arr,
            # 'y': hgt_arr,
            'y': dis_arr,
            'x': hgt_arr,
            'max_value': _max,
            'min_value': _min,
            'title': title
        })


@radar_api.route(API_GET_P_DETECT, methods=['POST', 'GET'])
def p_detect_cal():
    data = request.get_json()
    pfa = data['pfa']
    sigma = data['sigma']
    try:
        res = modules.radarCal.get_detect_p(pfa, sigma)
    except Exception as e:
        return json.dumps({
            'code': -1,
            'msg': '训练出错,' + str(e)
        })
    return json.dumps({
        'code': 0,
        'msg': '训练完成，结果表格已更新',
        'res': res,
        'table_entry': {
            'date': TimeUtil.current_time_str(),
            'pfa': pfa,
            'sigma': sigma,
            'pdetect': res
        }
    })
