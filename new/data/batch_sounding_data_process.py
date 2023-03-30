import os
import re

import numpy as np

from new.Util import DuctHeightUtil
from new.data.DataUtil import DataUtils


class SoundingDataProcess(DataUtils):


    SELECTED_ID = ['stn_58150', 'stn_59316', 'stn_58665', 'stn_59981', 'stn_58362', 'stn_57494', 'stn_54511',
                   'stn_58203', 'stn_58238', 'stn_58027', 'stn_58457', 'stn_58606', 'stn_58424', 'stn_59758']


    @staticmethod
    def batch_sounding_data_to_npy(root_path, des_path, filtered=False):
        if not filtered:
            _folders = DataUtils.get_all_file_names(root_path)
        else:
            _folders = SoundingDataProcess.SELECTED_ID
        for _folder in _folders:
            DataUtils.txt_file_to_npy(dir_=root_path + _folder, dest_=des_path + _folder, batch=True)
        print('batch_sounding_data_to_npy... Complete')


    @staticmethod
    def batch_interpolation_4_height(root_path, dest_path, avg_num=5, method='hypsometric', debug=False):
        """
        高度插值，根据前五个探测点的温度和压强，根据 hypsometric 或 barometric 求值，
        找到平均差值，然后填补当天空余的测量高度
        """
        _folders = DataUtils.get_all_file_names(root_path)
        for _sub_folder in _folders:
            os.makedirs(dest_path + _sub_folder, exist_ok=True)
            _sub_folders = DataUtils.get_all_file_names(root_path + _sub_folder)
            for _f in _sub_folders:
                _file_name = root_path + _sub_folder + '/' + _f
                _dest_name = dest_path + _sub_folder + '/' + _f
                _dataset = np.load(_file_name, allow_pickle=True)
                valid_cnt = 0
                valid_sum = 0
                for _ in range(avg_num):
                    e = _dataset[_]
                    real_hgt = e['HGNT']
                    real_temp = e['TEMP']
                    real_pres = e['PRES']
                    if real_hgt is None or real_temp is None or real_pres is None:
                        continue
                    else:
                        real_pres = real_pres * 0.1  # hPa to kPa
                        cal_hgt = DuctHeightUtil.cal_height_with_p_and_t(real_pres, real_temp, method=method)
                        gap = real_hgt - cal_hgt  # 真实值在前，即后面计算出来的值要 - gap
                        valid_sum += gap
                        valid_cnt += 1
                if valid_cnt == 0:
                    print('batch_interpolation_4_height... Valid_cnt = 0 for file {}'.format(_f))
                    continue

                offset = valid_sum / valid_cnt

                for _ in range(_dataset.size):
                    e = _dataset[_]
                    if e['HGNT'] is None:
                        if e['TEMP'] is not None and e['PRES'] is not None:
                            cal_hgt = DuctHeightUtil.cal_height_with_p_and_t(e['PRES'] * 0.1, e['TEMP'], method=method)
                            hgt = round(cal_hgt + offset, 1)
                            if 0 < _ < _dataset.size - 1:
                                if _dataset[_ + 1]['HGNT'] is not None and _dataset[_ - 1]['HGNT'] is not None:
                                    if hgt < _dataset[_ - 1]['HGNT'] or hgt > _dataset[_ + 1]['HGNT']:
                                        if debug:
                                            print('batch_interpolation_4_height... Invalid interpolation.'
                                                  'Setting val to avg. File: {}'.format(_f))
                                        hgt = (_dataset[_ - 1]['HGNT'] + _dataset[_ + 1]['HGNT']) / 2

                            e['HGNT'] = hgt
                        else:
                            print('batch_interpolation_4_height... Cannot interpolation for file {}'.format(_f))
                np.save(_dest_name, _dataset)
        print('batch_interpolation_4_height... Complete')


    @staticmethod
    def get_proper_sounding_data_info(root_path, n_heights=10, wrt=False):
        """
        由于探空点位太高，所以要找一些点位低的探测点
        取第一天的数据即可
        """
        _folders = DataUtils.get_all_file_names(root_path)
        # _file_name = root_path + '{}/{}_2020-01-01_00UTC.txt'
        stations = []
        for _f in _folders:
            if _f.find('.') >= 0:  # 可以修改一下逻辑。
                continue
            _folder_name = root_path + '{}/'.format(_f)
            _file_name = _folder_name + os.listdir(_folder_name)[0]  # 拿第一个 txt 文件
            with open(_file_name, mode='r') as _file:
                line = _file.readlines()
                pos_line = 2
                try:
                    location = re.findall('(?<=<H3>).*?(?=</H3>)', line[1])[0]
                except IndexError as e:
                    print(f'Error processing file {_f} with err: {e}')
                    location = ''
                    pos_line = 1
                lat = float(re.findall('(?<=Latitude: ).*?(?= Longitude:)', line[pos_line])[0])
                lng = float(re.findall('(?<=Longitude:).*?(?=</I>)', line[pos_line])[0])
                header = line[5]

                col_index = DataUtils.get_heading_idx_for_sounding_txt(header)
                col = 2
                height_idx_l = col_index[col - 1]
                height_idx_r = col_index[col] + 1

                heights = []

                _start = 8
                _ = _start
                while _ < len(line) - 1 and _ < _start + n_heights:
                    _val = line[_][height_idx_l: height_idx_r].strip()  # 取高度值
                    _ += 1
                    heights.append(float(_val) if _val else -1)

                station = DataUtils.Station(_id=_f, _lat=lat, _lon=lng, _location=location, _heights=heights)
                stations.append(station)

        stations = sorted(stations)
        print(stations)

        if wrt:
            # 写 excel
            wb, ws, output_name = DataUtils.excel_writer_prepare(header=['站点id', '位置', 'lat', 'lng'],
                                                                 output_name='station_height.xlsx')
            for s in stations:
                line = [s.id, s.location, s.lat, s.lon]
                for h in s.heights:
                    line.append(h)
                ws.append(line)

            wb.save(filename=output_name)

        return stations


if __name__ == '__main__':
    # print(SoundingDataProcess.get_proper_sounding_data_info('../data/sounding/', n_heights=10, wrt=True))
    SoundingDataProcess.batch_interpolation_4_height('../data/sounding_processed/', '../data/sounding_processed_hgt/')