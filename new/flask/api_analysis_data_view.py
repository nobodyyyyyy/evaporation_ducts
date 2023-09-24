from flask import request, Blueprint
import json

from new.Util.TimeUtil import TimeUtil
from new.flask.SupportedSingletons import SupportedSingletons

data_view_api = Blueprint('analysisDataView', __name__)

prefix = '/data/analysis'

API_GET_BASIC_INFO = prefix + '/init'
API_GET_SINGLE_DAY_DATA = prefix + '/fetch-single-date'
API_GET_RANGE_DATA = prefix + '/fetch-date-range'

# 单例列表
modules = SupportedSingletons()

mapping_eng = ['omega', 'q', 'skt', 'slp', 'sst', 'temp', 'u10m', 'uwind', 'v10m', 'vwind', 'zg']
mapping_cn = ['垂直速度', '比湿度', '地表温度', '平均海平面气压', '海平面温度', '温度', '10米U风分量', 'U风',
              '10米V风分量', 'V风', '重力加速度']
mapping_eng_to_cn = {}
for _ in range(len(mapping_eng)):
    mapping_eng_to_cn[mapping_eng[_]] = mapping_cn[_]


@data_view_api.route(API_GET_BASIC_INFO)
def init_info():
    """
    查询界面打开时调用，获取支持展示的：d
    1）再分析资料来源
    2）不同来源下的类型
    3）不同来源下类型的时间范围
    :return:
    """
    return json.dumps({
        'code': 0,
        'mapping_eng': mapping_eng,
        'mapping_cn': mapping_cn,
        'level': [1, 5, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
        'need_show_level': ['omega', 'q', 'temp', 'uwind', 'vwind', 'zg'],
        'data': [
            {
                'source': 'NOAA',
                'types': ['omega', 'q', 'skt', 'slp', 'sst', 'temp', 'u10m', 'uwind', 'v10m', 'vwind', 'zg'],
                'omega': {'desp': '垂直速度', 'date_from': '2018-01-01', 'date_to': '2018-01-31'},
                'q': {'desp': '比湿度', 'date_from': '2018-01-01', 'date_to': '2018-01-31'},
                'skt': {'desp': '地表温度', 'date_from': '2018-01-01', 'date_to': '2018-01-31'},
                'slp': {'desp': '平均海平面气压', 'date_from': '2018-01-01', 'date_to': '2018-01-31'},
                'sst': {'desp': '海平面温度', 'date_from': '2018-01-01', 'date_to': '2018-01-31'},
                'temp': {'desp': '温度', 'date_from': '2018-01-01', 'date_to': '2018-01-31'},
                'u10m': {'desp': '10米U风分量', 'date_from': '2018-01-01', 'date_to': '2018-01-31'},
                'uwind': {'desp': 'U风', 'date_from': '2018-01-01', 'date_to': '2018-01-31'},
                'v10m': {'desp': '10米V风分量', 'date_from': '2018-01-01', 'date_to': '2018-01-31'},
                'vwind': {'desp': 'V风', 'date_from': '2018-01-01', 'date_to': '2018-01-31'},
                'zg': {'desp': '重力加速度', 'date_from': '2018-01-01', 'date_to': '2018-01-31'},
            },
            {
                'source': 'EAR5',
                'types': ['omega', 'q', 'slp', 'sst', 'temp', 'u10m', 'uwind', 'v10m', 'vwind', 'zg'],
                'omega': {'desp': '垂直速度', 'date_from': '2020-01-01', 'date_to': '2020-12-31'},
                'q': {'desp': '比湿度', 'date_from': '2020-01-01', 'date_to': '2020-12-31'},
                'slp': {'desp': '平均海平面气压', 'date_from': '2020-01-01', 'date_to': '2020-12-31'},
                'sst': {'desp': '海平面温度', 'date_from': '2020-01-01', 'date_to': '2020-12-31'},
                'temp': {'desp': '温度', 'date_from': '2020-01-01', 'date_to': '2020-12-31'},
                'u10m': {'desp': '10米U风分量', 'date_from': '2020-01-01', 'date_to': '2020-12-31'},
                'uwind': {'desp': 'U风', 'date_from': '2020-01-01', 'date_to': '2020-12-31'},
                'v10m': {'desp': '10米V风分量', 'date_from': '2020-01-01', 'date_to': '2020-12-31'},
                'vwind': {'desp': 'V风', 'date_from': '2020-01-01', 'date_to': '2020-12-31'},
                'zg': {'desp': '重力加速度', 'date_from': '2020-01-01', 'date_to': '2020-12-31'},
            }
        ]
    })


@data_view_api.route(API_GET_SINGLE_DAY_DATA, methods=['POST', 'GET'])
def get_single_date_data():
    data = request.get_json()
    print(data)
    # lat_range = [i + 90 for i in data['lat_range']]
    lat_range = data['lat_range']
    lng_range = [i + 180 for i in data['lng_range']]
    type_ = data['type']
    source_name = data['source']
    source = modules.dataUtil.get_source_display(data['source'])
    timestamp = data['timestamp'] / 1000
    file_name = modules.dataUtil.get_nc_file_address(source, timestamp, type_, prefix='../data')
    level = data['level']
    dt = TimeUtil.timestamp_to_datetime(timestamp)
    y, m, d = TimeUtil.format_date_to_year_month_day(str(dt))

    ret = modules.dataUtil.get_support_data_single_date(year=y, month=m, type_=type_,
                                                        lan_s=lat_range[0], lan_e=lat_range[1],
                                                        lng_s=lng_range[0], lng_e=lng_range[1],
                                                        time_=timestamp, level=level,
                                                        file_name=file_name,
                                                        file_type=source)
    data, _max, _min = modules.dataUtil.gen_data_response_4_heatmap(ret.tolist())
    title = f'{source_name}-{mapping_eng_to_cn[type_]}'
    if level != '':
        title += f'-level{level}'
    title += f'-{str(dt)}-纬度区间[{lat_range[0]}~{lat_range[1]}]-经度区间[{lng_range[0]}-{lng_range[1]}]'
    return json.dumps({
        'code': 0,
        'lat': modules.dataUtil.fill_lat_lng(lat_range[0], lat_range[1]),
        'lng': modules.dataUtil.fill_lat_lng(lng_range[0], lng_range[1]),
        'data': data,
        'min_value': _min,
        'max_value': _max,
        'title': title
    })


@data_view_api.route(API_GET_RANGE_DATA, methods=['POST', 'GET'])
def get_range_data():
    data = request.get_json()
    print(data)
    lat = data['lat']
    lng = data['lng']
    type_ = data['type']
    source_name = data['source']
    source = modules.dataUtil.get_source_display(data['source'])
    timestamp_arr = data['timestamp']
    timestamp_s = timestamp_arr[0] / 1000
    timestamp_e = timestamp_arr[1] / 1000
    file_arr = modules.dataUtil.get_nc_file_addresses(source, timestamp_s, timestamp_e, type_, prefix='../data')
    level = data['level']
    dt_s = str(TimeUtil.timestamp_to_datetime(timestamp_s))
    dt_e = str(TimeUtil.timestamp_to_datetime(timestamp_e))
    y_s, m_s, d_s = TimeUtil.format_date_to_year_month_day(dt_s)
    y_e, m_e, d_e = TimeUtil.format_date_to_year_month_day(dt_e)
    ret = modules.dataUtil.get_support_data_range(year_start=y_s, month_start=m_s,
                                                  year_end=y_e, month_end=m_e, type_=type_, lan=lat,
                                                  lng=lng, time_start=timestamp_s, time_end=timestamp_e, level=level,
                                                  file_arr=file_arr, file_type=source)
    axis, ret, _max, _min = modules.dataUtil.gen_data_response_4_linechart(ret, dt_s, dt_e)
    title = f'{source_name}-{mapping_eng_to_cn[type_]}'
    if level != '':
        title += f'-level{level}'
    title += f'-纬度[{lat}]-经度[{lng}]-时间范围[{dt_s} - {dt_e}]'
    return json.dumps({
        'code': 0,
        'time_axis': axis,
        'data': ret,
        'min_value': _min,
        'max_value': _max,
        'title': title
    })
