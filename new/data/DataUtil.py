import bisect
import os
import time
import warnings

import numpy as np
import netCDF4 as nc
from numpy import ndarray
from openpyxl import Workbook

from new.Util.TimeUtil import TimeUtil


class DataUtils:

    _instance = None
    _exist = False

    FILE_TYPE_NOAA = 1
    FILE_TYPE_EAR5 = 2


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
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


    def __init__(self):
        if not DataUtils._exist:
            DataUtils._exist = True
            self.nc_dataset_cache = {}
            self.primary_dataset_cache = {}


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
        return dir_.split('/')[-1]


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
            print('txt_file_to_npy Complete.')


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
    def get_idx_for_val_pos(arr:ndarray, val, is_reverse=False, is_strict=False):
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
    def inner_get_idx(arr:ndarray, val, is_strict=False):
        n = len(arr)
        idx = bisect.bisect(arr, val)
        if arr[idx] == val:
            return idx
        if is_strict:
            return -1
        if idx >= n:
            # 考虑 arr 足够长
            return idx - 1
        else:
            # 取近邻
            # 要知道如果没找到，bisect 默认取右值
            if idx == 0:
                return idx
            elif arr[idx] - val > val - arr[idx - 1]:
                return idx - 1
            else:
                return idx


    @staticmethod
    def get_support_data(year:int, month:int, type_:str, lan:float, lng:float, time_, level=-1,
                         file_name='', file_type=FILE_TYPE_EAR5):
        """
        获取辅助数据，主要是 nc 文件内容
        :param month:
        :param year:
        :param type_: omega, q, skt, slp, sst, temp, u10m, uwind, v10m, vwind, zg
        :param lan: 纬度
        :param lng: 经度
        :param time_: 标准的【豪秒】级别时间戳
        :param level: air pressure level 只能取特定的取值，见文档
        :param file_name: 如果需要直接指定文件名，传入该项
        :param file_type: era5 和 noaa 文件的 key 可能不同
        :return: 所需 data
        """
        month = TimeUtil.format_month_or_day(month)

        if file_type == DataUtils.FILE_TYPE_EAR5:
            lat_desp = 'latitude'
            lon_desp = 'longitude'
            lat_idx_reverse = True
            nc_timestamp = TimeUtil.time_millis_2_nc_timestamp(time_)
        elif file_type == DataUtils.FILE_TYPE_NOAA:
            lat_desp = 'lat'
            lon_desp = 'lon'
            lat_idx_reverse = False
            nc_timestamp = TimeUtil.time_millis_2_noaa_timestamp(time_)
        else:
            print('get_support_data... file type [{}] not supported.'.format(file_type))
            return -1

        if file_name.strip() == '':
            # 没指定的一律读 era5 daily
            # 12/9/2022 ERA5 文件迁移及格式修改
            file = './ERA5_daily/{}/{}.{}-{}.daily.nc'.format(type_, type_, year, month)
        else:
            file = file_name
        inst = DataUtils()
        if file not in inst.nc_dataset_cache.keys():
            dataset = nc.Dataset(file)
            inst.nc_dataset_cache[DataUtils.get_file_name(file)] = dataset
        else:
            dataset = inst.nc_dataset_cache[DataUtils.get_file_name(file)]
        if dataset is None:
            print('get_support_data... Could not load dataset for file: {}'.format(file))
            return

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

        if type_ in ['skt', 'slp', 'sst', 'u10m', 'v10m']:
            # data with shape (time, latitude, longitude)
            return dataset.variables[type_][time_idx][lat_idx][lng_idx]
        elif type_ in ['omega', 'q', 'temp', 'uwind', 'vwind', 'zg']:
            # data with shape (time, level, latitude, longitude)
            if level == -1:
                print('Request level')
                return -1
            levels = np.array(dataset.variables['level'][:])
            level_idx = DataUtils.get_idx_for_val_pos(levels, level, is_strict=False)  # todo 大气压是否要严格相等呢？
            return dataset.variables[type_][time_idx][level_idx][lat_idx][lng_idx]



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
    DataUtils.txt_file_to_npy('./sounding/stn_52533/stn_52533_2020-01-01_00UTC.txt', './tes')
    pass
