import bisect
import os
import re
import time
import warnings
from datetime import datetime

import numpy as np
import netCDF4 as nc
import pandas as pd
from numpy import ndarray
from openpyxl import Workbook

from new.Util import DuctHeightUtil
from new.Util.TimeUtil import TimeUtil


class DataUtils:
    _instance = None
    _exist = False

    FILE_TYPE_NOAA = 1
    FILE_TYPE_EAR5 = 2
    FILE_TYPE_EAR5_2021_ODD = 3

    class Station:

        def __init__(self, _id, _lat, _lon, _location, _heights):
            self.id = _id
            self.lat = _lat
            self.lon = _lon
            self.location = _location
            self.heights = _heights

        def __lt__(self, other):
            if len(self.heights) == 0:
                return True
            elif len(other.heights) == 0 or self.heights[0] == 1:
                return False
            elif other.heights[0] == 1:
                return True
            return self.heights[0] < other.heights[0]

        def __repr__(self):
            return f'[{self.id}] {self.location}. lat: {self.lat}, lon: {self.lon}. heights: {self.heights}\n'

    def __new__(cls, *args, **kwargs):
        """
        有一说一，Util 还整个单例就很不合理命名的
        :param args:
        :param kwargs:
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not DataUtils._exist:
            DataUtils._exist = True
            self.nc_dataset_cache = {}
            self.primary_dataset_cache = {}

    @staticmethod
    def get_source_display(source_str, year=2020):
        if source_str == 'NOAA':
            return DataUtils.FILE_TYPE_NOAA
        elif source_str == 'EAR5' and year == 2021:
            return DataUtils.FILE_TYPE_EAR5_2021_ODD
        elif source_str == 'EAR5':
            return DataUtils.FILE_TYPE_EAR5
        else:
            return -1

    @staticmethod
    def type_rename(type_):
        if type_ == 'slp':
            type_ = 'msl'
        elif type_ == 'omega':
            type_ = 'w'
        elif type_ == 'temp':
            type_ = 't'
        elif type_ == 'u10m':
            type_ = 'u10'
        elif type_ == 'v10m':
            type_ = 'v10'
        elif type_ == 'uwind':
            type_ = 'u'
        elif type_ == 'vwind':
            type_ = 'v'
        elif type_ == 'zg':
            type_ = 'z'
        return type_

    @staticmethod
    def get_origin_file_address(station_id, date, prefix="."):
        """
        拿探空资料位置
        :param prefix: 前置位置
        :param date: 2020-01-01 like
        :param station_id:  stn_xxxxx
        :return:
        """
        return f'{prefix}/all_sounding_processed/{station_id}/{station_id}_{date}_00UTC.npy'

    @staticmethod
    def get_duct_info_file_address(station_id, prefix="../height_model"):
        """
        拿波导高度资料位置
        :param station_id:
        :param prefix:
        :return:
        """
        return f'{prefix}/merged/{station_id}.xlsx'

    @staticmethod
    def get_origin_file_addresses(station_id, data_from, date_to, prefix="."):
        """
        拿探空资料位置集合【顺序】
        :param station_id: stn_xxxxx
        :param data_from: 2020-01-01 like
        :param date_to: 2020-01-01 like
        :param prefix: 前置位置
        :return:
        """
        ret_files = []
        dates = pd.date_range(start=data_from, end=date_to)
        ret_dates = []
        for d in dates:
            date = f'{d.year}-{TimeUtil.format_month_or_day(d.month)}-{TimeUtil.format_month_or_day(d.day)}'
            f = f'{prefix}/all_sounding_processed/{station_id}/{station_id}_{date}_00UTC.npy'

            if os.path.exists(f):
                ret_files.append(f)
                ret_dates.append(date)
        return ret_files, ret_dates

    @staticmethod
    def get_nc_file_address(source_code, timestamp, type_name, prefix="."):
        if type(source_code) is str:
            source_code = DataUtils.get_source_display(source_code)
        if source_code == -1:
            print('DataUtil... get_file_address source_code unknown')
        ret = ''

        dt = TimeUtil.timestamp_to_datetime(timestamp)
        y, m, d = TimeUtil.format_date_to_year_month_day(str(dt))
        _year = str(y)
        _month = TimeUtil.format_month_or_day(m)

        if source_code == DataUtils.FILE_TYPE_NOAA:
            ret = f'{prefix}/AEM/{type_name}.{_year}-01.daily.nc'
        elif source_code == DataUtils.FILE_TYPE_EAR5:
            ret = f'{prefix}/ERA5_daily/{type_name}/{type_name}.{_year}-{_month}.daily.nc'
        # todo 2021 odd EAR5
        return ret

    @staticmethod
    def get_nc_file_addresses(source_code, timestamp_s, timestamp_e, type_name, prefix="."):
        """
        为周期数据拿文件数组
        :param timestamp_e:
        :param timestamp_s:
        :param source_code:
        :param type_name:
        :param prefix:
        :return:
        """
        if type(source_code) is str:
            source_code = DataUtils.get_source_display(source_code)
        if source_code == -1:
            print('DataUtil... get_file_address source_code unknown')
        ret = []
        dt_s = TimeUtil.timestamp_to_datetime(timestamp_s)
        dt_e = TimeUtil.timestamp_to_datetime(timestamp_e)
        y_s, m_s, d_s = TimeUtil.format_date_to_year_month_day(str(dt_s))
        y_e, m_e, d_e = TimeUtil.format_date_to_year_month_day(str(dt_e))
        year_s = str(y_s)
        year_e = str(y_e)
        # month_s = TimeUtil.format_month_or_day(m_s)
        # month_e = TimeUtil.format_month_or_day(m_e)

        # todo 目前没有需要处理的年份信息

        if source_code == DataUtils.FILE_TYPE_NOAA:
            # NOAA 的数据目前是一个文件夹就能包含的
            ret.append(f'{prefix}/AEM/{type_name}.{year_s}-01.daily.nc')
        elif source_code == DataUtils.FILE_TYPE_EAR5:
            for m in range(m_s, m_e + 1):
                ret.append(f'{prefix}/ERA5_daily/{type_name}/{type_name}.{year_s}-{TimeUtil.format_month_or_day(m)}.daily.nc')

        # todo 2021 odd EAR5
        return ret

    @staticmethod
    def get_all_file_names(path, recursive=False, filter_end=None):
        _ret = []
        if recursive:
            # 原子到具体文件路径，递归检查所有子文件夹
            for path, file_dir, files in os.walk(path):
                for file_name in files:
                    # print(os.path.join(path, file_name).replace('\\', '/'))  # 当前循环打印的是当前目录下的所有文件
                    _name = os.path.join(path, file_name).replace('\\', '/')
                    if filter_end is None or _name.endswith(filter_end):
                        _ret.append(_name)
                # for _dir in file_dir:
                #     print(os.path.join(path, _dir).replace('\\', '/'))  # 当前打印的是当前目录下的文件目录
        else:
            # 只拿一层，但不管你是不是文件夹
            _temp = os.listdir(path)
            for _name in _temp:
                if filter_end is None or _name.endswith(filter_end):
                    _ret.append(_name)
        return _ret

    @staticmethod
    def get_file_name(dir_):
        try:
            ret = dir_.split('/')[-1]
        except Exception as e:
            ret = dir_
            print('DataUtil... [Warning] get_file_name input: {}'.format(dir_))
        return ret

    @staticmethod
    def fill_lat_lng(start, end):
        ret = [start]
        while start != end + 1:
            start += 1
            ret.append(start)
        return ret

    @staticmethod
    def gen_data_response_4_heatmap(data):
        """
        根据 echarts 的热力图要求生成对应回包【再分析数据】
        :param data: 二维热力图数据
        :return:
        """
        ret = []
        # lat_n = data.shape[0]
        # lng_n = data.shape[1]
        _min = 10000000
        _max = 0
        try:
            lat_n = len(data)
            lng_n = len(data[0])
        except Exception as e:
            print('DataUtil... gen_data_response_4_heatmap... Error {}'.format(e))
            lat_n = 0
            lng_n = 0
            _min = 0
            _max = 0
        for i in range(0, lat_n):
            for j in range(0, lng_n):
                ret.append([i, j, data[i][j]])
                if data[i][j] != 0.:
                    _max = max(_max, data[i][j])
                    _min = min(_min, data[i][j])
        return ret, _max, _min

    @staticmethod
    def gen_data_response_4_linechart(data, time_s, time_e):
        """
        根据 echarts 的折线图要求生成对应回包【再分析数据】
        :return:
        """
        time_axis = []
        _min = 10000000
        _max = 0
        time_tmp = pd.date_range(time_s, time_e)
        for e in time_tmp:
            time_axis.append(e.strftime('%Y-%m-%d'))
        for e in data:
            if e != 0:
                _max = max(_max, e)
                _min = min(_min, e)
        # if int(_min) - 1 != 0:
        #     _min = int(_min) - 1
        # _max = int(_max) + 1
        return time_axis, data, _max, _min

    @staticmethod
    def excel_writer_prepare(header, output_name='', title_name='result'):
        # wrt
        wb = Workbook()
        ws = wb.active
        ws.title = title_name

        if output_name == '':
            output_name = 'output'
        if output_name.split('.')[-1] != 'xlsx' and output_name.split('.')[-1] != 'xls':
            output_name += '.xlsx'
        if header is not None:
            ws.append(header)
        return wb, ws, output_name

    @staticmethod
    def get_heading_idx_for_sounding_txt(header):
        # todo 有负数符号的情况会有 bug!
        # eg. 52323 站点
        col_index = [0]
        _l = -1
        processing = False
        for _r in range(len(header)):
            if header[_r] == ' ' and _l == -1:
                continue
            elif header[_r] != ' ' and not processing:
                _l = _r
                processing = True
            elif header[_r] == ' ' and processing:
                processing = False
                col_index.append(_r)
                _l = _r
        col_index.append(_r)
        return col_index

    @staticmethod
    def get_lat_and_lon_4_stn_data(_str):
        """
        传入 stn 格式的经纬度行，提取经纬度
        """
        lat = float(re.findall('(?<=Latitude: ).*?(?= Longitude:)', _str)[0])
        lng = float(re.findall('(?<=Longitude:).*?(?=</I>)', _str)[0])
        return lat, lng

    @staticmethod
    def get_location_4_stn_data(_str):
        return re.findall('(?<=<H3>).*?(?=</H3>)', _str)[0]

    @staticmethod
    def txt_file_to_npy(dir_, dest_, batch=False):
        """
        探空资料处理
        :param batch: 批处理
        :param dir_: 如果 batch，该项填文件夹目录
        :param dest_: 【重要】只用写需要生成的文件夹位置即可，文件名不用设置
        :return:
        """

        files = []

        if batch:
            files = DataUtils.get_all_file_names(path=dir_, recursive=False)
        else:
            files.append(dir_)

        os.makedirs(dest_, exist_ok=True)
        try:
            for f in files:
                with open('{}/{}'.format(dir_, f), mode='r') as file:
                    line = file.readlines()
                    # generate name
                    output = dest_ + '/' + f.split('.')[-2] + '.npy'

                    # bugfix: 12/9/2022 yrt 原本代码无法支持数据缺失和数据长度异常、过大等问题
                # 由于数据是右对齐的，所以取 header 的最右边的列，可以拿到对应的数据右边界
                header = line[5]
                col_index = DataUtils.get_heading_idx_for_sounding_txt(header)

                lst = []
                entries = ["PRES", "HGNT", "TEMP", "DWPT", "RELH", "MIXR", "DRCT", "SPED", "THTA", "THTE", "THTV"]
                temp = dict.fromkeys(entries)
                _ = 8
                while _ < len(line) - 1:
                    for col in range(1, len(col_index)):
                        _key = entries[col - 1]
                        _val = line[_][col_index[col - 1]: col_index[col] + 1].strip()
                        temp[_key] = float(_val) if _val else None
                    _ += 1
                    lst.append(temp.copy())

                np.save(output, lst)
            try:
                print('txt_file_to_npy Complete for station {}'.format(dir_.split('/')[-1]))
            except:
                print('txt_file_to_npy Complete')
        except:
            print('txt_file_to_npy Error occurs. No record for current station {}'.format(dir_))
            # todo 需要手動刪除錯誤的文件夾！！！
            # try:
            #     os.removedirs(dest_)
            # except:
            #     print('txt_file_to_npy When error occurs, files still cannot be removed')

    @staticmethod
    def aem_data_to_npy(dir_, dest_):
        warnings.warn("Deprecated method", DeprecationWarning)
        try:
            store = []
            with open(dir_) as f:
                line = f.readlines()
                _ = 0
                while _ < len(line):
                    # decode header
                    header = line[_].split()
                    tmp_header = {'HEADREC_ID': header[0], 'YEAR': int(header[1]), 'MONTH': int(header[2]),
                                  'DAY': int(header[3]), 'HOUR': int(header[4]), 'RELTIME': int(header[5]),
                                  'NUMLEV': int(header[6]), 'P_SRC_or_NP_SRC': header[7],
                                  'LAT': header[8][0:2] + '.' + header[8][2:],
                                  'LON': header[9][0:2] + '.' + header[9][2:], 'data': []}  # fixme 只适用于2位经纬度
                    n = tmp_header['NUMLEV']
                    for idx in range(_ + 1, _ + n + 1):
                        # 我们并不关心 A、B
                        data_ = line[idx]
                        data_ = data_.replace('A', ' ')
                        data_ = data_.replace('B', ' ')
                        arr = data_.split()
                        arr.insert(1, arr[0][1])
                        arr[0] = arr[0][0]
                        replace_list = ['-9999', '-8888']
                        arr = ['' if i in replace_list else i for i in arr]
                        tmp_header['data'].append(arr)
                    _ += (n + 1)
                    store.append(tmp_header)

                np.save(dest_, store)
        except Exception as e:
            print(e)

    @staticmethod
    def get_idx_for_val_pos(arr: ndarray, val, is_reverse=False, is_strict=False):
        """
        在 nc 文件中的 lan 或 lng 中找到 val 值对应的下标 idx
        nc 文件的 time 也是类似的调用该方法
        :param val:
        :param arr:
        :param is_reverse:
        :param is_strict: 是否严格查找
        """
        n = len(arr)
        if is_reverse:
            reverse_arr = np.flip(arr)
            idx = DataUtils.inner_get_idx(reverse_arr, val, is_strict)
            return n - idx - 1
        else:
            return DataUtils.inner_get_idx(arr, val, is_strict)

    @staticmethod
    def inner_get_idx(arr: ndarray, val, is_strict=False):
        n = len(arr)
        idx = bisect.bisect_left(arr, val)
        try:
            if arr[idx] == val:
                return idx
        except Exception as e:
            print('inner_get_idx... out of bounds ? for idx {}'.format(idx))
            if idx > 0:
                return idx - 1
        if is_strict:
            return -1
        if idx >= n:
            # 考虑 arr 足够长
            return idx - 1
        else:
            # 取近邻
            # 要知道如果没找到，bisect 默认取右值
            # if idx == 0:
            #     return idx
            # elif arr[idx] - val > val - arr[idx - 1]:
            #     return idx - 1
            # else:
            #     return idx
            return idx

    @staticmethod
    def get_support_data(year: int, month: int, type_: str, lan: float, lng: float, time_, level=-1,
                         file_name='', file_type=FILE_TYPE_EAR5, search_bounds=5):
        """
        获取辅助数据，主要是 nc 文件内容 【再分析资料主要获取方法】
        这个方法是很久之前写的，主要为的是拿单个数据
        :param month:
        :param year:
        :param type_: omega, q, skt, slp, sst, temp, u10m, uwind, v10m, vwind, zg
        :param lan: 纬度
        :param lng: 经度
        :param time_: 标准的【豪秒】级别时间戳
        :param level: air pressure level 只能取特定的取值，见文档
        :param file_name: 如果需要直接指定文件名，传入该项
        :param file_type: era5 和 noaa 文件的 key 可能不同
        :param search_bounds: 一个位置查找不到，在周围查找的数组下标界限
        :return: 所需 data
        """
        month = TimeUtil.format_month_or_day(month)

        if file_type == DataUtils.FILE_TYPE_EAR5:
            lat_desp = 'latitude'
            lon_desp = 'longitude'
            lat_idx_reverse = True
            nc_timestamp = TimeUtil.time_millis_2_nc_timestamp(time_)
        elif file_type == DataUtils.FILE_TYPE_NOAA:
            # 2023/9/15 使用 AEM 文件夹下文件 key 修改
            # lat_desp = 'lat'
            # lon_desp = 'lon'
            lat_desp = 'latitude'
            lon_desp = 'longitude'
            lat_idx_reverse = True
            # todo 为什么  NOAA 时间戳变了
            nc_timestamp = TimeUtil.time_millis_2_nc_timestamp(time_)
        elif file_type == DataUtils.FILE_TYPE_EAR5_2021_ODD:
            lat_desp = 'lat'
            lon_desp = 'lon'
            lat_idx_reverse = True
            nc_timestamp = TimeUtil.time_millis_2_nc_timestamp(time_)
        else:
            print('get_support_data... file type [{}] not supported.'.format(file_type))
            return -999

        if file_name.strip() == '':
            # 没指定的一律读 era5 daily
            # 12/9/2022 ERA5 文件迁移及格式修改
            file = './ERA5_daily/{}/{}.{}-{}.daily.nc'.format(type_, type_, year, month)
        else:
            file = file_name
        inst = DataUtils()
        if DataUtils.get_file_name(file) not in inst.nc_dataset_cache.keys():
            dataset = nc.Dataset(file)
            inst.nc_dataset_cache[DataUtils.get_file_name(file)] = dataset
        else:
            dataset = inst.nc_dataset_cache[DataUtils.get_file_name(file)]
        if dataset is None:
            print('get_support_data... Could not load dataset for file: {}'.format(file))
            return -999

        lats = np.array(dataset.variables[lat_desp][:])
        lngs = np.array(dataset.variables[lon_desp][:])
        times = np.array(dataset.variables['time'][:])
        # find corresponding idx for lan, lng, time
        lat_idx = DataUtils.get_idx_for_val_pos(lats, lan, is_reverse=lat_idx_reverse)
        lng_idx = DataUtils.get_idx_for_val_pos(lngs, lng)
        try:
            time_idx = DataUtils.get_idx_for_val_pos(times, nc_timestamp)
        except IndexError as e:
            # q 数据集时间是有缺失的我真无语住了
            # 这种情况就（合理地）认为，使用传入的时间戳 time 转换为临近的日作为 time_idx
            print(e)
            year_ = time.localtime(time_).tm_year
            month_ = time.localtime(time_).tm_mon
            day = time.localtime(time_).tm_mday
            hour = time.localtime(time_).tm_hour
            if hour >= 12 and day != 1 and day != TimeUtil.get_day_sum(year_, month_):
                time_idx = day  # 加一天
            else:
                time_idx = day - 1
        type_ = DataUtils.type_rename(type_)
        if type_ in ['skt', 'slp', 'sst', 'u10m', 'v10m', 'msl', 'u10', 'v10']:
            # data with shape (time, latitude, longitude)
            data_type = 1
            val = dataset.variables[type_][time_idx][lat_idx][lng_idx]

        elif type_ in ['omega', 'q', 'temp', 'uwind', 'vwind', 'zg', 'w', 't', 'u', 'v', 'z']:
            # data with shape (time, level, latitude, longitude)
            data_type = 2
            if level == -1:
                print('Request level')
                return -999
            levels = np.array(dataset.variables['level'][:])
            level_idx = DataUtils.get_idx_for_val_pos(levels, level, is_strict=False)  # todo 大气压是否要严格相等呢？
            val = dataset.variables[type_][time_idx][level_idx][lat_idx][lng_idx]
        else:
            print('get_support_data... type_ is not supported')
            return -999

        # 2023/4/3 如果一个位置拿不到，尝试拿隔壁位置
        if val is np.ma.masked:
            for _ in range(1, search_bounds + 1):
                x_pos = [lat_idx - _, lat_idx, lat_idx + _]
                y_pos = [lng_idx - _, lng_idx, lng_idx + _]
                for lat_ in x_pos:
                    for lng_ in y_pos:
                        try:
                            val = dataset.variables[type_][time_idx][lat_][lng_] if data_type == 1 \
                                else dataset.variables[type_][time_idx][level_idx][lat_][lng_]
                        except Exception as e:
                            pass
                        if val is not np.ma.masked:
                            print(
                                '[Warning] get_sst... While searching sst pos, program reached bound {}. '
                                'Lat: {} Lon: {} '.format(_, lan, lng))
                            return val

        if val is np.ma.masked:
            print('[Error] get_support_data... Can not fetch sst because there is no record to find. '
                  'Lat: {} Lon: {} '.format(lan, lng))
            return -999
        return val

    @staticmethod
    def get_support_data_single_date(year: int, month: int, type_: str, lan_s: float, lan_e: float,
                                     lng_s: float, lng_e: float, time_, level=-1, file_name='',
                                     file_type=FILE_TYPE_EAR5,):
        """
        获取单天的某个再分析资料的热力图数据
        和上面方法高度重复，懒得合并了
        :return:
        """
        month = TimeUtil.format_month_or_day(month)

        if file_type == DataUtils.FILE_TYPE_EAR5:
            lat_desp = 'latitude'
            lon_desp = 'longitude'
            lat_idx_reverse = True
            nc_timestamp = TimeUtil.time_millis_2_nc_timestamp(time_)
        elif file_type == DataUtils.FILE_TYPE_NOAA:
            # 2023/9/15 使用 AEM 文件夹下文件 key 修改
            # lat_desp = 'lat'
            # lon_desp = 'lon'
            lat_desp = 'latitude'
            lon_desp = 'longitude'
            lat_idx_reverse = True
            # todo 为什么  NOAA 时间戳变了
            nc_timestamp = TimeUtil.time_millis_2_nc_timestamp(time_)
        elif file_type == DataUtils.FILE_TYPE_EAR5_2021_ODD:
            lat_desp = 'lat'
            lon_desp = 'lon'
            lat_idx_reverse = True
            nc_timestamp = TimeUtil.time_millis_2_nc_timestamp(time_)
        else:
            print('get_support_data_single_date... file type [{}] not supported.'.format(file_type))
            return -999

        if file_name.strip() == '':
            # 没指定的一律读 era5 daily
            # 12/9/2022 ERA5 文件迁移及格式修改
            file = './ERA5_daily/{}/{}.{}-{}.daily.nc'.format(type_, type_, year, month)
        else:
            file = file_name
        inst = DataUtils()
        if DataUtils.get_file_name(file) not in inst.nc_dataset_cache.keys():
            dataset = nc.Dataset(file)
            inst.nc_dataset_cache[DataUtils.get_file_name(file)] = dataset
        else:
            dataset = inst.nc_dataset_cache[DataUtils.get_file_name(file)]
        if dataset is None:
            print('get_support_data_single_date... Could not load dataset for file: {}'.format(file))
            return -999

        lats = np.array(dataset.variables[lat_desp][:])
        lngs = np.array(dataset.variables[lon_desp][:])
        times = np.array(dataset.variables['time'][:])
        # find corresponding idx for lan, lng, time
        lat_s_idx = DataUtils.get_idx_for_val_pos(lats, lan_s, is_reverse=lat_idx_reverse)
        lng_s_idx = DataUtils.get_idx_for_val_pos(lngs, lng_s)
        lat_e_idx = DataUtils.get_idx_for_val_pos(lats, lan_e, is_reverse=lat_idx_reverse)
        lng_e_idx = DataUtils.get_idx_for_val_pos(lngs, lng_e)
        if lat_idx_reverse:
            _ = lat_s_idx
            lat_s_idx = lat_e_idx
            lat_e_idx = _
        try:
            time_idx = DataUtils.get_idx_for_val_pos(times, nc_timestamp)
        except IndexError as e:
            # q 数据集时间是有缺失的我真无语住了
            # 这种情况就（合理地）认为，使用传入的时间戳 time 转换为临近的日作为 time_idx
            print(e)
            year_ = time.localtime(time_).tm_year
            month_ = time.localtime(time_).tm_mon
            day = time.localtime(time_).tm_mday
            hour = time.localtime(time_).tm_hour
            if hour >= 12 and day != 1 and day != TimeUtil.get_day_sum(year_, month_):
                time_idx = day  # 加一天
            else:
                time_idx = day - 1
        type_ = DataUtils.type_rename(type_)
        if type_ in ['skt', 'slp', 'sst', 'u10m', 'v10m', 'msl', 'u10', 'v10']:
            # data with shape (time, latitude, longitude)
            temp = np.ma.filled(dataset.variables[type_][time_idx], fill_value=0)
            val = temp[lat_s_idx: lat_e_idx + 1, lng_s_idx: lng_e_idx + 1]

        elif type_ in ['omega', 'q', 'temp', 'uwind', 'vwind', 'zg', 'w', 't', 'u', 'v', 'z']:
            # data with shape (time, level, latitude, longitude)
            if level == -1:
                print('Request level')
                return None
            levels = np.array(dataset.variables['level'][:])
            level_idx = DataUtils.get_idx_for_val_pos(levels, level, is_strict=False)  # todo 大气压是否要严格相等呢？
            temp = np.ma.filled(dataset.variables[type_][time_idx][level_idx], fill_value=0)
            val = temp[lat_s_idx: lat_e_idx, lng_s_idx: lng_e_idx]
        else:
            print('get_support_data_single_date... type_ is not supported')
            return None

        return val

    @staticmethod
    def get_support_data_range(year_start: int, month_start: int,
                               year_end: int, month_end: int, type_: str, lan: float,
                               lng: float, time_start, time_end, level=-1, file_arr=None,
                               file_type=FILE_TYPE_EAR5, ):
        """
        获取时间段内的某个再分析资料的折线图数据
        和上面方法高度重复，懒得合并了
        :return:
        """
        if file_arr is None:
            file_arr = []
        if file_type == DataUtils.FILE_TYPE_EAR5:
            lat_desp = 'latitude'
            lon_desp = 'longitude'
            lat_idx_reverse = True
            start_time = TimeUtil.time_millis_2_nc_timestamp(time_start)
            end_time = TimeUtil.time_millis_2_nc_timestamp(time_end)
        elif file_type == DataUtils.FILE_TYPE_NOAA:
            # 2023/9/15 使用 AEM 文件夹下文件 key 修改
            # lat_desp = 'lat'
            # lon_desp = 'lon'
            lat_desp = 'latitude'
            lon_desp = 'longitude'
            lat_idx_reverse = True
            # todo 为什么 NOAA 时间戳又变了？
            start_time = TimeUtil.time_millis_2_nc_timestamp(time_start)
            end_time = TimeUtil.time_millis_2_nc_timestamp(time_end)
        elif file_type == DataUtils.FILE_TYPE_EAR5_2021_ODD:
            lat_desp = 'lat'
            lon_desp = 'lon'
            lat_idx_reverse = True
            start_time = TimeUtil.time_millis_2_nc_timestamp(time_start)
            end_time = TimeUtil.time_millis_2_nc_timestamp(time_end)
        else:
            print('get_support_data_range... file type [{}] not supported.'.format(file_type))
            return -999

        month_s = TimeUtil.format_month_or_day(month_start)
        month_e = TimeUtil.format_month_or_day(month_end)
        inst = DataUtils()
        type_ = DataUtils.type_rename(type_)
        # val = np.array()
        ret = []
        for file_idx in range(0, len(file_arr)):
            file = file_arr[file_idx]
            if DataUtils.get_file_name(file) not in inst.nc_dataset_cache.keys():
                dataset = nc.Dataset(file)
                inst.nc_dataset_cache[DataUtils.get_file_name(file)] = dataset
            else:
                dataset = inst.nc_dataset_cache[DataUtils.get_file_name(file)]
            if dataset is None:
                print('get_support_data_range... Could not load dataset for file: {}'.format(file))
                return -999

            lats = np.array(dataset.variables[lat_desp][:])
            lngs = np.array(dataset.variables[lon_desp][:])
            times = np.array(dataset.variables['time'][:])
            # find corresponding idx for lan, lng, time
            lat_idx = DataUtils.get_idx_for_val_pos(lats, lan, is_reverse=lat_idx_reverse)
            lng_idx = DataUtils.get_idx_for_val_pos(lngs, lng)

            # 如果 idx 是 0，取开始时间 ~ 结尾
            # 如果 idx 是最后一个，取开始位置 ~ 结束时间
            # 其他情况全取
            if len(file_arr) == 1:
                # 只有一个文件，正常处理
                time_idx_s = DataUtils.get_idx_for_val_pos(times, start_time)
                time_idx_e = DataUtils.get_idx_for_val_pos(times, end_time)
                val = dataset.variables[type_][time_idx_s:time_idx_e + 1]
            elif file_idx == 0:
                time_idx_s = DataUtils.get_idx_for_val_pos(times, start_time)  # todo 去除了 q 的时间判断，因为后来都没发生
                val = dataset.variables[type_][time_idx_s:]
            elif file_idx == len(file_arr) - 1:
                time_idx_e = DataUtils.get_idx_for_val_pos(times, end_time)
                val = dataset.variables[type_][:time_idx_e + 1]
            else:
                # 不需要额外时间 idx 信息
                val = dataset.variables[type_][:]

            val = np.ma.filled(val, fill_value=0)

            if type_ in ['skt', 'slp', 'sst', 'u10m', 'v10m', 'msl', 'u10', 'v10']:
                # data with shape (time, latitude, longitude)
                ret += val[:, lat_idx, lng_idx].tolist()
            elif type_ in ['omega', 'q', 'temp', 'uwind', 'vwind', 'zg', 'w', 't', 'u', 'v', 'z']:
                # data with shape (time, level, latitude, longitude)
                if level == -1:
                    print('Request level')
                    return -999
                levels = np.array(dataset.variables['level'][:])
                level_idx = DataUtils.get_idx_for_val_pos(levels, level, is_strict=False)  # todo 大气压是否要严格相等呢？
                ret += val[:, level_idx, lat_idx, lng_idx].tolist()
            else:
                print('get_support_data_range... type_ is not supported')
                return None

        return ret

    @staticmethod
    def generate_ref_and_h(data, axis=None):
        """
        输入data，生成其他算法需要的廓线和高度数据
        :param data:
        :param axis: axis = [Temp, Press, Hum, H]
        :return:
        """
        if axis is None:
            axis = [0, 1, 2, 3]
        h = np.zeros((data.shape[0], 1))
        ref = np.zeros((data.shape[0], 1))
        for i in range(0, data.shape[0]):
            h[i][0] = data[i][axis[3]]
            ref[i][0] = DuctHeightUtil.atmospheric_refractive_index_M(data[i][axis[0]],
                                                                      data[i][axis[1]],
                                                                      data[i][axis[2]],
                                                                      data[i][axis[3]])
        return ref, h


if __name__ == '__main__':
    # DataUtils.aem_data_to_npy('./AEM/AEM00041217-data.txt', './AEM/AEM00041217-data.npy')
    # read_dictionary = np.load('./AEM/AEM00041217-data.npy', allow_pickle=True)
    # pass
    # DataUtils.get_support_data(2018, 1, 'q', 42.2, 42.2, 1514880149, 150)

    # DataUtils.txt_file_to_npy(r'E:\test_data\test_data\test1.txt',r'E:\test_data\test_data\test1.npy')
    # f= np.load(r'E:\test_data\test_data\test1.npy', allow_pickle=True)
    # print(f)

    # DataUtils.txt_file_to_npy('./CN/test1.txt', './CN/shantou.npy')
    # DataUtils.txt_file_to_npy('./CN/test2.txt', './CN/haikou.npy')

    # DataUtils.txt_file_to_npy('../data/test_2022_12_02/sounding_data/stn_59758',
    #                           '../data/test_2022_12_02/sounding_data/stn_59758_processed',
    #                           batch=True)
    # DataUtils.txt_file_to_npy('./sounding/stn_52533/stn_52533_2020-01-01_00UTC.txt', './tes')

    # DataUtils.get_support_data_single_date(2020, 1, 'sst', 42.2, 47, 40, 48, 1578903004)

    DataUtils.get_nc_file_address(1, 1578903004, 'a')
    pass
