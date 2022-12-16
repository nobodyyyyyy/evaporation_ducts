import os

import numpy as np
from openpyxl import Workbook

from new.Util.DuctHeightUtil import kelvins2degrees, atmospheric_refractive_index_M, get_duct_height
from new.Util.MathUtil import MathUtil
from new.data.DataUtil import DataUtils
from new.Util.TimeUtil import TimeUtil
from new.height_model.models.babin import babin_duct_height
from new.height_model.models.nps import nps_duct_height
from new.height_model.models.pj2height import pj2height
from new.height_model.models.sst import evap_duct_SST


class HeightCal:

    _instance = None
    _exist = False

    SST_NOT_FOUND = -999

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
            self.models = [nps_duct_height, babin_duct_height, evap_duct_SST, pj2height]
            self.sst_cache = {}


    def get_sst(self, year:int, month:int, day:int, lan:float, lng:float):
        if (year, month, lan, lng) in self.sst_cache.keys():
            return self.sst_cache[(year, month, lan, lng)]
        time_ = TimeUtil.to_time_millis(year, month, day, 0, 0, 0)
        month = TimeUtil.format_month_or_day(month)
        sst_file = '../data/ERA5_daily/sst/sst.{}-{}.daily.nc'.format(year, month)
        try:
            sst_kelvins = DataUtils.get_support_data(year, month, 'sst', lan, lng, time_, file_name=sst_file)
        except FileNotFoundError as e:
            print('get_sst... no such file to read sst info. Errinfo: {}'.format(e))
            return HeightCal.SST_NOT_FOUND
        self.sst_cache[(year, month, lan, lng)] = kelvins2degrees(sst_kelvins)
        return kelvins2degrees(sst_kelvins)


    def get_data(self, data_dir, year, month, day, lan, lng, sst=-999):
        if sst == HeightCal.SST_NOT_FOUND:
            sst = self.get_sst(year, month, day, lan, lng)
        name = DataUtils.get_file_name(data_dir)
        if name in self.dataset_cache.keys():
            dataset = self.dataset_cache[name]
        else:
            dataset = np.load(data_dir, allow_pickle=True)
            self.dataset_cache[name] = dataset
        return dataset, sst


    def cal_height(self, data_dir, model, year, month, day, lan, lng, sst=0, nrows=1):
        dataset, sst = self.get_data(data_dir, year, month, day, lan, lng, sst)
        if sst == HeightCal.SST_NOT_FOUND:
            return None
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
            res.append(self.cal_height_with_data(e, sst, func))
            _ += 1
        return res


    def cal_height_with_data(self, e, sst, model, stable_check=False):
        """
        有数据和海面温度的时候，就可以计算高度
        :param model: 模型名称或函数
        :param e: npy 格式的处理好的文件
        :param sst: 海面温度
        :param stable_check: 检查是否稳定
        :return: 波导高度
        """
        if type(model) is str:
            if model not in self.model_entry.keys():
                print('cal_height_with_data... U')
            model = self.models[self.model_entry[model]]
        is_stable = False
        t = e['TEMP']  # 气温
        eh = e['RELH']  # 相对湿度
        u = e['SPED']  # 风速
        p = e['PRES']  # 压强
        h = e['HGNT']  # 测量高度
        if t is None or eh is None or u is None or p is None:
            print('cal_height_with_data... data incomplete: [{}, {}, {}, {}, {}, {}]'.format(t, eh, sst, u, p, h))
            return None
        try:
            if stable_check:
                res, is_stable = model(t, eh, sst, u, p, h, stable_check)
            else:
                res = model(t, eh, sst, u, p, h)
        except Exception as e:
            print('cal_height_with_data... model error: {}'.format(e))
            return None
        if stable_check:
            return res, is_stable
        return res


    def cal_and_record_all_models(self, data_dir, year, month, day, lan, lng, output_name='', nrows=1):
        wb = Workbook()
        ws = wb.active
        ws.title = 'result'
        header = ['气温', '相对湿度', '海温', '风速', '压强', '测量高度']
        res = []
        sst = self.get_sst(year, month, day, lan, lng)
        for model_name in self.model_entry.keys():
            header.append(model_name)
            tmp_res = self.cal_height(data_dir, model_name, year, month, day, lan, lng, sst=sst, nrows=nrows)
            res.append(tmp_res)

        # wrt
        if output_name == '':
            output_name = 'output'
        if output_name.split('.')[-1] != 'xlsx' and output_name.split('.')[-1] != 'xls':
            output_name += '.xlsx'

        name = DataUtils.get_file_name(data_dir)
        input_data = self.dataset_cache[name]  # you can always find target in cache

        ws.append(header)
        for i in range(nrows):
            e = input_data[i]
            line = [e['TEMP'], e['RELH'], sst, e['SPED'], e['PRES'], e['HGNT']]
            for j in range(len(self.model_entry.keys())):
                line.append(res[j][i])
            ws.append(line)
        wb.save(filename=output_name)


    def batch_cal_and_record_all_models(self, data_dir, lan, lng, output_name='', stable_check=False):
        """
        批处理模型高度计算，全模型。
        其实可以和 cal_and_record_all_models 结合，但没必要引入非必要的耦合
        """
        wb = Workbook()
        ws = wb.active
        ws.title = 'result'
        header = ['日期', '气温', '相对湿度', '海温', '风速', '压强', '测量高度']
        if not stable_check:
            header += list(self.model_entry.keys())
        else:
            for model_name in list(self.model_entry.keys()):
                header += [model_name, '稳定性']

        files = []
        for file_name in os.listdir(data_dir):
            files.append(file_name)

        res = []  # [[date, input..., results...], [], [], ...]

        # cal each
        for file in files:
            date_str = file.split('_')[2]
            _year, _month, _day = TimeUtil.format_date_to_year_month_day(date_str)
            _file = data_dir + '/' + file  # real dir

            dataset, sst = self.get_data(_file, _year, _month, _day, lan, lng)
            ele = dataset[0]
            cur_res = [date_str, ele['TEMP'], ele['RELH'], sst, ele['SPED'], ele['PRES'], ele['HGNT']]
            if sst == HeightCal.SST_NOT_FOUND:
                res.append(cur_res)
                continue
            for _model in self.model_entry.keys():
                try:
                    if stable_check:
                        height, is_stable = self.cal_height_with_data(ele, sst, _model, stable_check)
                        cur_res.append(height)
                        cur_res.append(is_stable)
                    else:
                        height = self.cal_height_with_data(ele, sst, _model)
                        cur_res.append(height)
                except Exception as err:
                    print('batch_cal_and_record_all_models... error for file: {}\n Info: {}'.format(file, err))
                    cur_res.append(None)
            print('batch_cal_and_record_all_models... appending {}'.format(cur_res))
            res.append(cur_res)


        # wrt
        if output_name == '':
            output_name = 'output'
        if output_name.split('.')[-1] != 'xlsx' and output_name.split('.')[-1] != 'xls':
            output_name += '.xlsx'
        ws.append(header)
        for l in res:
            ws.append(l)
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


    def sensitivity_analyze(self, data_dir, model, year, month, day, lan, lng, sst=0, output_name=''):
        """
        单模型，敏感性分析
        或全模型（model == 'all'）
        """
        dataset, sst = self.get_data(data_dir, year, month, day, lan, lng, sst)
        res = {}
        func_arr = []
        header = ['气温', '相对湿度', '海温', '风速', '压强', '测量高度']
        if model.lower() != 'all':
            if model in self.model_entry.keys():
                func_arr.append(self.models[self.model_entry[model]])
                header.append(model)
            else:
                print('cal_height... Unexpected model: {}'.format(model))
                return
        else:
            func_arr = self.models
            for model_name in self.model_entry.keys():
                header.append(model_name)
        # 只关心第一条
        e = dataset[0]
        t = e['TEMP']  # 气温
        eh = e['RELH']  # 相对湿度
        u = e['SPED']  # 风速
        p = e['PRES']  # 压强
        h = e['HGNT']  # 测量高度
        if t is None or eh is None or u is None or p is None:
            print('sensitivity_analyze... data incomplete: [{}, {}, {}, {}, {}, {}]'.format(t, eh, sst, u, p, h))
            return
        new_data = self.disturbance_prepare(t, eh, sst, u, p, h)

        tmp_res = []
        for r in new_data:
            model_res = []
            for _model in func_arr:
                try:
                    # r 的顺序和 disturbance_prepare 传参顺序保持一致
                    height = _model(r[0], r[1], r[2], r[3], r[4], r[5])
                except Exception as e:
                    # 有什么错我们都不会终止
                    print('sensitivity_analyze... t={}, eh={}, u={}, p={}, h={} error:{}'.format(t, eh, u, p, h, e))
                    height = None
                model_res.append(height)
            tmp_res.append(model_res)

        # wrt
        wb = Workbook()
        ws = wb.active
        ws.title = 'result'

        if output_name == '':
            output_name = 'output'
        if output_name.split('.')[-1] != 'xlsx' or output_name.split('.')[-1] != 'xls':
            output_name += '.xlsx'

        # record
        ws.append(header)

        assert len(new_data) == len(tmp_res)
        for _ in range(len(new_data)):
            ws.append(new_data[_] + tmp_res[_])

        wb.save(filename=output_name)

        return res


    @staticmethod
    def disturbance_prepare(t, eh, sst, u, p, h, lowers=None, uppers=None, gaps=None, round_first=False):
        """
        扰动数据准备
        论文里面是修改 u风速{1,4,7}、t气温[-5,5]、eh湿度[50, 100, GAP:5]
        初步扰动约束：
        u   gap:1   range: ±3
        t   gap:1   range: ±5
        eh  gap:5%  range: 50~100% (±25%?)
        """
        ret = []
        if round_first:
            res = MathUtil.round(u, t, eh)
            u, t, eh = res[0], res[1], res[2]
        if lowers is None:
            lowers = [-3, -5, -25]
        if uppers is None:
            uppers = [3, 5, 25]
        if gaps is None:
            gaps = [1, 1, 5]
        for u_gap in range(lowers[0], uppers[0] + gaps[0], gaps[0]):
            cur_u = MathUtil.add(u, u_gap)
            if cur_u < 0:
                continue
            for t_gap in range(lowers[1], uppers[1] + gaps[1], gaps[1]):
                cur_t = MathUtil.add(t, t_gap)
                for eh_gap in range(lowers[2], uppers[2] + gaps[2], gaps[2]):
                    cur_eh = MathUtil.add(eh, eh_gap)
                    if cur_eh > 100 or cur_eh < 0:
                        continue
                    # 生成扰动数据
                    # 当前仅针对 t eh u 扰动
                    ret.append([cur_t, cur_eh, sst, cur_u, p, h])

        return ret


if __name__ == '__main__':
    c = HeightCal()
    # c.sensitivity_analyze('../data/CN/haikou.npy', 'all', 2021, 11, 29, 20, 110.250, output_name='sensi')
    # print(c.disturbance_prepare(23, 71, 23.231, 4.1, 1021, 65))
    # c.cal_and_record_all_models('../data/CN/haikou.npy', 2021, 11, 29, 20.000, 110.250, 'haikou',nrows=1
    c.batch_cal_and_record_all_models('../data/test_2022_12_02/sounding_data/stn_59758_processed', 20.000, 110.250,
                                      output_name='haikou_all_stable.xlsx', stable_check=True)
    # todo 似乎有一个更高分辨率的sst可以去尝试拿取
    # c.cal_and_record_all_models('../data/CN/shantou.npy', 2021, 11, 29, 23.350, 116.670, 'shantou')
    # print(c.cal_real_height('../data/CN/haikou.npy'))
    pass


