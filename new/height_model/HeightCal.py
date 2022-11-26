import numpy as np
from matplotlib import pyplot as plt
from openpyxl import Workbook

from new.Util.DuctHeightUtil import kelvins2degrees, atmospheric_refractive_index_M, get_duct_height
from new.data.DataUtil import DataUtils
from new.Util.TimeUtil import TimeUtil
from new.height_model.models.babin import babin_duct_height
from new.height_model.models.nps import nps_duct_height
from new.height_model.models.pj2height import pj2height
from new.height_model.models.sst import evap_duct_SST


class HeightCal:

    _instance = None
    _exist = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


    def __init__(self):
        if not HeightCal._exist:
            HeightCal._exist = True
            self.dataset_cache = {}
            self.model_entry = {
                'nps': 0,
                'babin': 1,
                'sst': 2,
                'pj': 3
            }
            # t, RH, ts, u, P
            self.models = [nps_duct_height, babin_duct_height, evap_duct_SST, pj2height]


    @staticmethod
    def get_sst(year:int, month:int, day:int, lan:float, lng:float, file_name=''):
        time_ = TimeUtil.to_time_millis(year, month, day, 0, 0, 0)
        sst_kelvins = DataUtils.get_support_data(year, month, 'sst', lan, lng, time_, file_name=file_name)
        return kelvins2degrees(sst_kelvins)


    def cal_height(self, data_dir, model, year, month, day, lan, lng, sst=0, nrows=1):
        if sst == 0:
            sst = HeightCal.get_sst(year, month, day, lan, lng, file_name='../data/CN/ERA5_hourly_00_sst_2021.nc')
        name = DataUtils.get_file_name(data_dir)
        if name in self.dataset_cache.keys():
            dataset = self.dataset_cache[name]
        else:
            dataset = np.load(data_dir, allow_pickle=True)
            self.dataset_cache[name] = dataset

        if model in self.model_entry.keys():
            func = self.models[self.model_entry[model]]
        else:
            print('cal_height... Unexpected model: {}'.format(model))
            return

        # todo 仅适用于海温不变的情况（因为现在一个文件只有一天），后续有改变可以扩展
        # 22/11/25 fix 波导高度的模型计算只用高度最低的一条数据即可
        res = []
        _ = 0
        for e in dataset:
            if _ == nrows:
                break
            t = e['TEMP']  # 气温
            eh = e['RELH']  # 相对湿度
            u = e['SPED']  # 风速
            p = e['PRES']  # 压强
            h = e['HGNT']  # 测量高度
            if t is None or eh is None or u is None or p is None:
                print('cal_height... data incomplete: [{}, {}, {}, {}, {}]'.format(t, eh, sst, u, p, h))
                res.append(None)
                continue
            res.append(func(t, eh, sst, u, p, h))
            _ += 1
        return res


    def cal_and_record_all_models(self, data_dir, year, month, day, lan, lng, output_name='', nrows=1):
        wb = Workbook()
        ws = wb.active
        ws.title = 'result'
        header = ['气温', '相对湿度', '海温', '风速', '压强', '测量高度']
        res = []
        sst = HeightCal.get_sst(year, month, day, lan, lng, file_name='../data/CN/ERA5_hourly_00_sst_2021.nc')
        for model_name in self.model_entry.keys():
            header.append(model_name)
            tmp_res = self.cal_height(data_dir, model_name, year, month, day, lan, lng, sst=sst, nrows=nrows)
            res.append(tmp_res)

        # wrt
        if output_name == '':
            output_name = 'output'
        if output_name.split('.')[-1] != 'xlsx' or output_name.split('.')[-1] != 'xls':
            output_name += '.xlsx'

        name = DataUtils.get_file_name(data_dir)
        input_data = self.dataset_cache[name]  # you can always find target in cache

        ws.append(header)
        # 22/11/25 fix 波导高度的模型计算只用高度最低的一条数据即可
        for i in range(nrows):
            e = input_data[i]
            line = [e['TEMP'], e['RELH'], sst, e['SPED'], e['PRES'], e['HGNT']]
            for j in range(len(self.model_entry.keys())):
                line.append(res[j][i])
            ws.append(line)
        wb.save(filename=output_name)


    def cal_real_height(self, data_dir):
        """
        计算不同高度的折射率，画出廓线，找到拐点，得到真实的大气波导高度
        """
        name = DataUtils.get_file_name(data_dir)
        if name in self.dataset_cache.keys():
            dataset = self.dataset_cache[name]
        else:
            dataset = np.load(data_dir, allow_pickle=True)
            self.dataset_cache[name] = dataset
        # todo 一个 data_dir 对应某一时间的文件，后续文件格式改变需要修改
        _Ms = []  # M 折射率
        _Zs = []  # Z 高度
        for e in dataset:
            t = e['TEMP']  # 气温
            eh = e['RELH']  # 相对湿度
            p = e['PRES']  # 压强
            h = e['HGNT']  # 测量高度
            if t is None or eh is None or h is None or p is None:
                print('cal_real_height... data incomplete: [{}, {}, {}, {}]'.format(t, eh, p, h))
                continue
            _Ms.append(atmospheric_refractive_index_M(t, p, eh, h))
            _Zs.append(h)
        return get_duct_height(_Ms, _Zs, caller='cal_real_height')


if __name__ == '__main__':
    c = HeightCal()
    c.cal_height('../data/CN/haikou.npy', 'sst', 2021, 11, 29, 20, 110.250, nrows=1)
    # c.cal_and_record_all_models('../data/CN/haikou.npy', 2021, 11, 29, 20.000, 110.250, 'haikou',nrows=5)
    # c.cal_and_record_all_models('../data/CN/shantou.npy', 2021, 11, 29, 23.350, 116.670, 'shantou')
    # print(c.cal_real_height('../data/CN/haikou.npy'))
    pass


