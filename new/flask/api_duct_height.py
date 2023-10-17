import numpy as np
import pandas as pd
from flask import Blueprint, request
import json

from new.Util.MathUtil import MathUtil
from new.Util.TimeUtil import TimeUtil
from new.data.batch_sounding_data_process import SoundingDataProcess
from new.flask.SupportedSingletons import SupportedSingletons
from new.height_model.models.babin import babin_duct_height
from new.height_model.models.nps import nps_duct_height
from new.height_model.models.pj2height import pj2height
from new.height_model.models.sst import evap_duct_SST

duct_height_api = Blueprint('ductHeight', __name__)

prefix = '/height'

API_CAL_HEIGHT = prefix + '/cal'

API_INIT_ENTRY = prefix + '/init_entry'
API_GET_HEIGHT_TABLE = prefix + '/fetch-height-raw'
API_GET_HEIGHT_GRAPH = prefix + '/fetch-height-graph'

# header_values = ['日期', '气温(°C)', '相对湿度(%)', '海温(°C)', '风速(m/s)', '压强(hPa)', '测量高度(m)',
#                  'nps(m)', 'babin(m)', 'liuli2.0(m)', 'pj(m)']
# header_keys = ['date', 'temp', 'relh', 'sst', 'speed', 'pressure', 'height', 'nps', 'babin', 'liuli2.0', 'pj']

header_values = ['日期', '气温(°C)', '相对湿度(%)', '海温(°C)', '风速(m/s)', '压强(hPa)', '测量高度(m)',
                 'nps(m)', 'byc(m)', 'mgb(m)', 'pj(m)']
header_keys = ['date', 'temp', 'relh', 'sst', 'speed', 'pressure', 'height', 'nps', 'byc', 'mgb', 'pj']

excel_header = ['日期', '气温', '相对湿度', '海温', '风速', '压强', '测量高度',
                'nps', 'babin', 'liuli2.0', 'pj']
col_map = []
for _ in range(len(header_keys)):
    col_map.append({'eng': header_keys[_], 'cn': header_values[_]})

# 单例列表
modules = SupportedSingletons()

stations = {}

model_entry = {
    'nps': 0,
    'babin': 1,
    'liuli': 2,
    'pj': 3
}
models = [nps_duct_height, babin_duct_height, evap_duct_SST, pj2height]


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
        if ele[0] not in SoundingDataProcess.SELECTED_ID:  # 过滤
            continue
        stations[ele[0]] = Station(ele[0], ele[2], ele[3], ele[1])


@duct_height_api.route(API_INIT_ENTRY, methods=['POST', 'GET'])
def init_entry():
    """
    查询界面打开时调用，获取支持展示的：
    1）站点【限制】信息数据  位置、經緯度等等
    2）站點數據的時間範圍，當前都是 2020~2021年，先不處理
    3) 表格表头
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
        'date_to': '2021-12-31',
        'predict_date_to': '2021-12-28',
        'map': col_map
    })


@duct_height_api.route(API_CAL_HEIGHT, methods=['POST', 'GET'])
def cal_height():
    data = request.get_json()
    print(data)
    t = float(data['temp'])  # 气温
    eh = float(data['relh'])  # 相对湿度
    u = float(data['speed'])  # 风速
    p = float(data['pressure'])  # 压强
    h = float(data['height'])  # 测量高度
    sst = float(data['sst'])  # 海面温度
    model_name = data['model']
    model = models[model_entry[model_name]]

    # 名称更变
    if model_name == 'babin':
        model_name = 'byc'
    if model_name == 'liuli':
        model_name = 'mgb'
    try:
        res = model(t, eh, sst, u, p, h)
        res = str(round(res, 2))
    except Exception as e:
        print('cal_height... Error: {}'.format(str(e)))
        return json.dumps({
            'code': -1,
            'msg': str(e),
            'hint_': '请检查参数输入'
        })
    return json.dumps({
        'code': 0,
        'res': res,
        'table_entry': {
            'date': TimeUtil.current_time_str(),
            'model': model_name,
            'temp': str(t),
            'relh': str(eh),
            'speed': str(u),
            'pressure': str(p),
            'height': str(h),
            'sst': sst,
            'res': res
        }
    })


@duct_height_api.route(API_GET_HEIGHT_TABLE, methods=['POST', 'GET'])
def get_raw_duct_height_data():
    """
    以表格形式返回波导高度数据【实际上就是excel对应行】
    :return:
    """
    data = request.get_json()
    date_from = data['date'][0]
    date_to = data['date'][1]
    _id = data['id']

    file_addr = modules.dataUtil.get_duct_info_file_address(station_id=_id, prefix='../height_model')
    dataset = pd.read_excel(file_addr)
    ret = []
    start = False
    for _, row in dataset.iterrows():
        cur_time = TimeUtil.str_to_timestamp(row['日期'])
        if not start:
            if date_from <= cur_time <= date_to:
                start = True
            else:
                continue
        else:
            # 文件是时间排序的
            if cur_time > date_to:
                break
        row_data = {'date': row['日期']}
        for idx in range(1, len(excel_header)):
            row_data[header_keys[idx]] = round(row[excel_header[idx]], 4)
        ret.append(row_data)

    return json.dumps({
        'code': 0,
        'data': ret
    })


@duct_height_api.route(API_GET_HEIGHT_GRAPH, methods=['POST', 'GET'])
def get_graph_duct_height_data():
    """
    以图形式返回波导高度数据【实际上就是excel对应行】
    :return:
    """
    data = request.get_json()
    date_from = data['date'][0]
    date_to = data['date'][1]
    _id = data['id']

    file_addr = modules.dataUtil.get_duct_info_file_address(station_id=_id, prefix='../height_model')
    dataset = pd.read_excel(file_addr)
    ret = [[],[],[],[]]
    header = ['nps', 'babin', 'liuli2.0', 'pj']
    start = False
    dates = []
    for _, row in dataset.iterrows():
        cur_time = TimeUtil.str_to_timestamp(row['日期'])
        if not start:
            if date_from <= cur_time <= date_to:
                start = True
            else:
                continue
        else:
            # 文件是时间排序的
            if cur_time > date_to:
                break
        dates.append(row['日期'])
        for _ in range(0, len(header)):
            ret[_].append(round(row[header[_]], 4))
        ret.append(ret)

    ret_series = []
    # 名称更变
    header = ['nps', 'byc', 'mgb', 'pj']
    for _ in range(len(header)):
        ret_series.append({
            'name': header[_],
            'type': 'line',
            'stack': 'Total',
            'data': ret[_]
        })

    s_date = TimeUtil.timestamp_to_datetime(date_from / 1000)
    e_date = TimeUtil.timestamp_to_datetime(date_to/ 1000)
    title = f'{_id}_{stations[_id].location}_[{s_date} ~ {e_date}]'

    return json.dumps({
        'code': 0,
        'title': title,
        'x': dates,
        'types': header,
        'series': ret_series
    })


init_api()
