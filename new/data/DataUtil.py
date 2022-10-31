import numpy as np
import netCDF4 as nc


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
    def get_support_data(year, month, type_, lan, lon, time, level=-1):
        """
        获取辅助数据，主要是 nc 文件内容
        :param month:
        :param year:
        :param type_: omega, q, skt, slp, sst, temp, u10m, uwind, v10m, vwind, zg
        :param lan: 纬度
        :param lon: 经度
        :param time: 标准的秒级别时间戳，本方法会做对应的转换
        :param level: air pressure level 只能取特定的取值，见文档
        :return: 所需 data
        """
        file = './AEM/{}.{}-{}.daily.nc'.format(type_, year, month)
        dataset = nc.Dataset(file)
        all_vars = dataset.variables.keys()
        all_vars_info = dataset.variables.items()


        if type_ in ['skt', 'slp', 'sst', 'u10m', 'v10m']:
            # data with shape (time, latitude, longitude)



            pass
        else:
            pass

        # lat = dataset.variables['latitude'][:]
        # print(lat.shape)
        # var_data = np.array(lat)
        # print(var_data)



if __name__ == '__main__':
    DataUtils.aem_data_to_npy('./AEM/AEM00041217-data.txt', './AEM/AEM00041217-data.npy')
    read_dictionary = np.load('./AEM/AEM00041217-data.npy', allow_pickle=True)
    pass
