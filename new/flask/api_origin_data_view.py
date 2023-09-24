import re

import pandas as pd
from flask import request, Blueprint
import json
import numpy as np

from new.Util.TimeUtil import TimeUtil
from new.flask.SupportedSingletons import SupportedSingletons

origin_data_view_api = Blueprint('originDataView', __name__)

# 单例列表
modules = SupportedSingletons()

prefix = '/data/origin'

API_GET_BASIC_INFO = prefix + '/init'
API_GET_DAILY_INFO = prefix + '/fetch-single'

API_GET_COMMON_HEADER = prefix + '/header'
API_GET_RANGE_INFO = prefix + '/fetch-range'

API_GET_SINGLE_DAY_ENTRY = prefix + '/single-day-entry'
API_GET_DAILY_SINGLE_INFO = prefix + '/fetch-single-daily'

stations = {}

keys = ['PRES', 'HGNT', 'TEMP', 'DWPT', 'RELH', 'MIXR',
        'DRCT', 'SPED', 'THTA', 'THTE', 'THTV']
names = ['大气压强 (hPa)', '高度 (m)', '温度 (°C)', '露点温度 (°C)', '相对湿度 (%)', '配合比 (g/kg)',
         '风向 (°)', '风速 (m/s)', '位温 (K)', '相当位温 (K)', '虚相当位温 (K)']
# todo 风速单位需要确认

key_name_map = []  # 傻逼命名
single_day_key_name_map = []
type_map = {}
for _ in range(0, len(keys)):
    key_name_map.append({'eng': keys[_], 'cn': names[_]})
    type_map[keys[_]] = names[_]
    if keys[_] != 'HGNT':
        single_day_key_name_map.append({'eng': keys[_], 'cn': names[_]})


class Station:

    def __init__(self, _id, _lat, _lon, _location):
        self.id = _id
        self.lat = _lat
        self.lon = _lon
        self.location = _location

    def __lt__(self, other):
        if self.id < other.id:
            return True
        else:
            return False

    def __repr__(self):
        return f'[{self.id}] {self.location}. lat: {self.lat}, lon: {self.lon}.\n'


def init_api(excel_dir='../data/station_info.xlsx'):
    # 讀 excel 初始化信息
    info = pd.read_excel(excel_dir)
    for ele in info.values:
        if ele[0] == '' or ele[1] == '' or np.isnan(ele[2]) or np.isnan(ele[3]):
            continue
        stations[ele[0]] = Station(ele[0], ele[2], ele[3], ele[1])


@origin_data_view_api.route(API_GET_BASIC_INFO)
def init_info():
    """
    查询界面打开时调用，获取支持展示的：
    1）站点信息数据  位置、經緯度等等
    2）站點數據的時間範圍，當前都是 2020~2021年，先不處理
    :return:
    """
    ids = []
    locations = []
    lats = []
    lngs = []

    for k in sorted(stations.keys()):
        station = stations[k]
        ids.append(station.id)
        locations.append(station.location)
        lats.append(station.lat)
        lngs.append(station.lon)

    return json.dumps({
        'code': 0,
        'ids': ids,
        'locations': locations,
        'lats': lats,
        'lngs': lngs,
        'date_from': '2020-01-01',
        'date_to': '2021-12-31'
    })


@origin_data_view_api.route(API_GET_COMMON_HEADER, methods=['POST', 'GET'])
def fetch_header():
    return json.dumps({
        'code': 0,
        'map': key_name_map
    })


@origin_data_view_api.route(API_GET_SINGLE_DAY_ENTRY, methods=['POST', 'GET'])
def fetch_single_day_entry():
    return json.dumps({
        'code': 0,
        'map': single_day_key_name_map
    })


@origin_data_view_api.route(API_GET_DAILY_INFO, methods=['POST', 'GET'])
def fetch_daily():
    data = request.get_json()
    print(data)
    date = data['date']
    _id = data['id']
    file = modules.dataUtil.get_origin_file_address(_id, date, prefix='../data')
    dataset = np.load(file, allow_pickle=True)
    return json.dumps({
        'code': 0,
        'cols_eng': keys,
        'cols_ch': names,
        'map': key_name_map,
        'data': dataset.tolist()
    })


@origin_data_view_api.route(API_GET_DAILY_SINGLE_INFO, methods=['POST', 'GET'])
def fetch_daily_single():
    """
    垂直风廓线
    :return:
    """
    data = request.get_json()
    print(data)
    date = data['date']
    _id = data['id']
    _type = data['type']
    file = modules.dataUtil.get_origin_file_address(_id, date, prefix='../data')
    dataset = np.load(file, allow_pickle=True)
    heights = []
    values = []
    _min = 1000000
    _max = -1000000
    height_min = 0
    height_max = -1
    for _ in range(dataset.size):
        if dataset[_]['HGNT'] is None or dataset[_][_type] is None:
            # 不记录
            continue
        heights.append(dataset[_]['HGNT'])
        values.append(dataset[_][_type])
        _min = min(_min, dataset[_][_type])
        _max = max(_max, dataset[_][_type])
        height_min = min(height_min, dataset[_]['HGNT'])
        height_max = max(height_max, dataset[_]['HGNT'])

    if _min == 1000000:
        _min = 0
        _max = 0

    title = f'{_id}_{stations[_id].location}_{date}_{type_map[_type]}-高度廓线'

    return json.dumps({
        'code': 0,
        'heights': heights,
        'values': values,
        'value_min_max': [_min, _max],
        'height_min_max': [height_min, height_max],
        'title': title,
        'unit': re.findall(r".*\((.*)\)", type_map[_type])[0],
        'cn_desp': type_map[_type]
    })


@origin_data_view_api.route(API_GET_RANGE_INFO, methods=['POST', 'GET'])
def fetch_range():
    data = request.get_json()
    print(data)
    date_from = data['date'][0]
    date_to = data['date'][1]
    _id = data['id']
    level = data['level']
    types = data['type']
    files, dates = modules.dataUtil.get_origin_file_addresses(_id, date_from, date_to, prefix='../data')

    # 探测点位置合法检测
    check_dataset = np.load(files[0], allow_pickle=True)
    if check_dataset.size < level:
        return json.dumps({
            'code': -1,
            'msg': f'探测点位过高。记载的最高探测点位为{check_dataset.size}'
        })
    level = level - 1
    ret_data = []
    for _ in types:
        ret_data.append([])

    for _f in range(len(files)):
        dataset = np.load(files[_f], allow_pickle=True)
        for _t in range(len(types)):
            type_name = types[_t]
            ret_data[_t].append(dataset[level][type_name])
    ret_series = []

    for _t in range(len(types)):
        # types 重命名，加上中文
        types[_t] = type_map[types[_t]] + ' ' + types[_t]
        ret_series.append({
            'name': types[_t],
            'type': 'line',
            'stack': 'Total',
            'data': ret_data[_t]
        })

    title = f'{_id}_{stations[_id].location}_[{date_from} ~ {date_to}]_点位[{level + 1}]'

    return json.dumps({
        'code': 0,
        'title': title,
        'x': dates,
        'types': types,
        'series': ret_series
    })


init_api()
