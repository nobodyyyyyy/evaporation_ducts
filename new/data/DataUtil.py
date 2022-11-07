import bisect
import time

import numpy as np
import netCDF4 as nc
from numpy import ndarray

from new.Util.TimeUtil import TimeUtil


class DataUtils:

    def __init__(self):
        pass

    @staticmethod
    def aem_data_to_npy(dir_, dest_):
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
    def get_support_data(year:int, month:int, type_:str, lan:float, lng:float, time_, level=-1):
        """
        获取辅助数据，主要是 nc 文件内容
        :param month:
        :param year:
        :param type_: omega, q, skt, slp, sst, temp, u10m, uwind, v10m, vwind, zg
        :param lan: 纬度
        :param lng: 经度
        :param time_: 标准的【豪秒】级别时间戳
        :param level: air pressure level 只能取特定的取值，见文档
        :return: 所需 data
        """
        month = TimeUtil.format_month_or_day(month)
        file = './AEM/{}.{}-{}.daily.nc'.format(type_, year, month)
        dataset = nc.Dataset(file)
        nc_timestamp = TimeUtil.time_millis_2_nc_timestamp(time_)
        lats = np.array(dataset.variables['latitude'][:])
        lngs = np.array(dataset.variables['longitude'][:])
        times = np.array(dataset.variables['time'][:])
        # find corresponding idx for lan, lng, time
        lat_idx = DataUtils.get_idx_for_val_pos(lats, lan, is_reverse=True)
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
    read_dictionary = np.load('./AEM/AEM00041217-data.npy', allow_pickle=True)
    pass
    # DataUtils.get_support_data(2018, 1, 'q', 42.2, 42.2, 1514880149, 150)
