import math
import re

import numpy as np

from new.Util import DuctHeightUtil
from new.Util.DrawUtil import DrawUtil
from new.Util.MathUtil import MathUtil
from new.data.DataUtil import DataUtils
# from data import SP
from new.height_model.HeightCal import HeightCal
from new.radar_model import SP
from new.radar_model import newspe
# from data import SPEE


class RadarCal:

    def __init__(self):
        # self.dataset_cache = {}
        pass

    @staticmethod
    def radar_data_pre_generate(ref, h):
        pre_data = np.zeros((5, ref.shape[1]))
        flag1 = np.zeros((400, ref.shape[1]))
        flag2 = np.zeros((400, ref.shape[1]))
        hh1 = np.zeros((400, ref.shape[1]))
        hh2 = np.zeros((400, ref.shape[1]))
        tidu1 = np.zeros((400, ref.shape[1]))
        tidu2 = np.zeros((400, ref.shape[1]))
        g1 = np.zeros((400, ref.shape[1]))
        g2 = np.zeros((400, ref.shape[1]))
        delm2 = np.zeros((400, ref.shape[1]))
        ddm1 = np.zeros((400, ref.shape[1]))
        cc1 = np.zeros((400, ref.shape[1]))
        ddm2 = np.zeros((400, ref.shape[1]))
        r = None
        for i in range(ref.shape[1]):
            if r is None:
                r = [np.gradient(ref[:, i], h[:, i])]
            else:
                r.append(np.gradient(ref[:, i], h[:, i]))
        r = np.array(r).transpose(1, 0)
        print(r.shape)
        # 折点计算
        # g1为波导点到非波导点高度，g2为非波导点到波导点高度，判断波导类型
        n = 0
        k = 0
        # 因为用法都是一条廓线，这里不在设置多条的情况
        for i in range(ref.shape[1]):
            for j in range(ref.shape[0] - 1):
                if r[j, i] <= 0 and r[j + 1, i] > 0 and r[0, i] < 0:
                    hh1[n, i] = h[j, i]
                    g1[n, i] = h[j, i]
                    ddm1[n, i] = ref[0, i] - ref[j, i]
                    tidu1[n, i] = ddm1[n, i] / hh1[n, i]
                    if g1[n, i] < 40:
                        flag1[n, i] = 1
                    else:
                        flag1[n, i] = 2
                    n += 1
                elif r[j + 1, i] <= 0 and r[j, i] > 0 and r[0, i] > 0:
                    delm2[k, i] = ref[j, i]
                    for ss in range(ref.shape[0] - j - 1):
                        if r[j + ss, i] <= 0 and r[j + ss + i, i] > 0:
                            if ref[j + ss, i] > ref[0, i]:
                                flag2[k, i] = 3
                                g2[k, i] = h[j, i]
                                ddm2[k, i] = ref[j + ss, i] - delm2[k, i]
                                for b in range(j - 1):
                                    if np.abs(h[b, i] - h[j, i]) < 0.1:
                                        hh2[k, i] = h[j + ss, i] - g2[k, i]
                                        tidu2[k, i] = ddm2[k, i] / hh2[k, i]
                        elif ref[j + ss, i] < ref[0, i]:
                            flag2[k, i] = 4
                            ddm2[k, i] = ref[j + ss, i] - delm2[k, i]
                            hh2[k, i] = h[j + ss, i] - h[j, i]
                            tidu2[k, i] = delm2[k, i] / g2[k, i]
                    cc1[k, i] = delm2[k, i] / g2[k, i]
                    k += 1
        for i in range(0, ref.shape[1]):
            for j in range(0, flag1.shape[0]):
                if (flag1[j][i] == 1):
                    # 发生蒸发波导
                    c1 = 0
                    zb = 0
                    for k in range(0, tidu1.shape[0]):
                        if (tidu1[k][i] != 0):
                            m = tidu1[k][i]
                            break
                    for k in range(0, hh1.shape[0]):
                        if (hh1[k][i] != 0):
                            zt = hh1[k][i]
                            q = zt
                            break
                    pre_data[0][i] = m
                    pre_data[1][i] = zt
                    pre_data[2][i] = c1
                    pre_data[3][i] = zb
                    pre_data[4][i] = q
                    break
                elif (flag1[j][i] == 2):
                    # 发生表面波导1
                    zb = 0
                    q = 0
                    c1 = 0
                    for k in range(0, tidu1.shape[0]):
                        if (tidu1[k][i] != 0):
                            m = tidu1[k][i]
                            break
                    for k in range(0, hh1.shape[0]):
                        if (hh1[k][i] != 0):
                            zt = hh1[k][i]
                            break
                    pre_data[0][i] = m
                    pre_data[1][i] = zt
                    pre_data[2][i] = c1
                    pre_data[3][i] = zb
                    pre_data[4][i] = q
                    break
            for j in range(0, flag2.shape[0]):
                if (flag2[j][i] == 3):
                    # 发生了悬空波导
                    m = 0
                    zt = 0
                    c1 = 0
                    zb = 0
                    q = 0
                    for k in range(0, tidu2.shape[0]):
                        if (tidu2[k][i] != 0):
                            m = tidu2[k][i]
                            break
                    for k in range(0, hh2.shape[0]):
                        if (hh2[k][i] != 0):
                            zt = hh2[k][i]
                            break
                    for k in range(0, cc1.shape[0]):
                        if (cc1[k][i] != 0):
                            c1 = cc1[k][i]
                            break
                    for k in range(0, g2.shape[0]):
                        if (g2[k][i] != 0):
                            zb = g2[k][i]
                            break
                    pre_data[0][i] = m
                    pre_data[1][i] = zt
                    pre_data[2][i] = c1
                    pre_data[3][i] = zb
                    pre_data[4][i] = q
                    break
                elif (flag2[j][i] == 4):
                    # 发生表面波导2
                    q = 0
                    c1 = 0
                    zb = 0
                    for k in range(0, tidu2.shape[0]):
                        if (tidu2[k][i] != 0):
                            m = tidu2[k][i]
                            break
                    for k in range(0, hh2.shape[0]):
                        if (hh2[k][i] != 0):
                            zt = hh2[k][i]
                            break
                    for k in range(0, g2.shape[0]):
                        if (g2[k][i] != 0):
                            zb = g2[k][i]
                            break
                    pre_data[0][i] = m
                    pre_data[1][i] = zt
                    pre_data[2][i] = c1
                    pre_data[3][i] = zb
                    pre_data[4][i] = q
                    break
        return pre_data

    def get_l_single(self, data_dir, radar_feq, antenna_height):
        """
        拿 L_{single}
        :param data_dir: npy 探空地址
        :param antenna_height: 天线高度
        :param radar_feq 雷达频率
        """
        # name = DataUtils.get_file_name(data_dir)
        # if name in self.dataset_cache.keys():
        #     dataset = self.dataset_cache[name]
        # else:
        #     dataset = np.load(data_dir, allow_pickle=True)
        #     self.dataset_cache[name] = dataset
        # # todo 一个 data_dir 对应某一时间的文件，后续文件格式改变需要修改
        # data_list = []
        # for e in dataset:
        #     t = e['TEMP']  # 气温
        #     eh = e['RELH']  # 相对湿度
        #     p = e['PRES']  # 压强
        #     h = e['HGNT']  # 测量高度
        #     if t is None or eh is None or h is None or p is None:
        #         if debug:
        #             print('get_l_single... data incomplete: [{}, {}, {}, {}]'.format(t, eh, p, h))
        #         continue
        #     data_list.append([t, p, eh, h])
        # data_list = np.array(data_list)
        # # data_list = np.load('../../Algorithm/snd.npy')
        # # data_list = data_list[:, [3, 0, 2, 1]]
        #
        # ref, h = DataUtils.generate_ref_and_h(data_list)
        #
        # pre_data = RadarCal.radar_data_pre_generate(ref, h)
        # pre_data[0][0], pre_data[1][0], pre_data[2][0], pre_data[3][0], pre_data[4][0] = \
        #     MathUtil.round(pre_data[0][0], pre_data[1][0], pre_data[2][0], pre_data[3][0], pre_data[4][0])
        # # 检测是否有波导发生
        # valid_f = False
        # for _ in range(5):
        #     if pre_data[_][0] != 0:
        #         valid_f = True
        # if not valid_f:
        #     print('get_l_single... Invalid. No ref and h found for input.')
        # else:
        #     print('get_l_single... Valid. {} {} {} {} {}'.format(pre_data[0][0], pre_data[1][0],
        #                                                          pre_data[2][0], pre_data[3][0], pre_data[4][0]))
        # 无论如何都计算并画图
        # m: 负折射指数, zt: 陷获层厚度, c1: 混合层斜率, zb: 陷获层底厚度, q: 蒸发波导厚度
        # loss = SP.spe(pre_data[0][0], pre_data[1][0], pre_data[2][0], pre_data[3][0], pre_data[4][0])
        # loss = SP.spe(0.137, 100.0, -0.043, 300, 0)
        # pre_data[0][0], pre_data[1][0], pre_data[2][0], pre_data[3][0], pre_data[4][0] = -53, 20, 0, 0.0, 70
        # loss = SP.spe(pre_data[0][0], pre_data[1][0], pre_data[2][0], pre_data[3][0], pre_data[4][0])
        # loss = np.array(loss)
        # # loss = np.flip(loss, axis=1)
        # # loss = np.flip(loss, axis=0)
        # # loss = loss.swapaxes(0, 1)
        inst = HeightCal('../data/sounding')  # 单例
        height, _ = inst.cal_real_height(data_dir)
        if height == 0:
            print('[Warning] get_l_single... Duct height is 0')
        loss = newspe.spee(height, radar_feq, antenna_height)
        # loss = newspe.spee(40, 9400, 8)
        return loss

    @staticmethod
    def get_Ts(f=8600, Pt=230, G=30, D0=60, Bn=769230, Ls=30, F0=5, sigma=20):
        # f:雷达频率(MHz)  [5600]
        # Pt:雷达峰值功率(KW) [论文写的是发射功率 Pt] [230]
        # G:天线增益(dB) [30]
        # D0:最小信噪比(dB) D0 为雷达信号的检测因子(最小可检测信噪比)  []  (60)
        # Bn:接收机带宽(MHz) 与脉宽 t 满足关系：Bn ≈ 1/ t  t=1.3μs  [769230]
        # Ls:系统综合损耗(dB) [] （30）
        # F0:接收机噪声系数(dB) []  (找不到，5)
        # sigma:目标散射截面(m*m) σ [0.2 / 20]
        # 输出lossflag 门限

        #   没用到的：Text11:目标高度(m)  Text6:发射仰角(deg)  Text5:波束宽度(deg)  Text3:天线高度(m)
        Lossflag = 135.43 + 10 * math.log10(Pt * sigma * f * f) - \
                   10 * math.log10(Bn) + \
                   2 * G - Ls - F0 - \
                   10 * math.log10(D0)
        Lossflag = Lossflag * 0.5
        return Lossflag

    # @staticmethod
    # def get_Ls(f, R, F):
    #     # Ls 电波在实际环境的单程传播损耗
    #     # f:雷达频率(MHz)
    #     # R:目标斜距(km)
    #     # F:传播因子(dB)
    #     # return Lfs - 20 * math.log10(F)
    #     # Ls 与大气传播环境(大气波导)密切相关,可利用电磁波传播的抛物数值方程计算得到.
    #     return 32.44 + 20 * math.log10(f) + 20 * math.log10(R) - 20 * math.log10(F)

    @staticmethod
    def is_detected(Ls, Ts):
        """
        Ls <= Ts 即目标能够被检测到
        """
        return Ls <= Ts

    @staticmethod
    def get_detected_field(Ls, Ts):
        """
        根据 Ls 矩阵和 Ts 信息获取最远传输距离曲线
        :return: 最远传输距离曲线 (1x高度)，每个高度对应一个最远距离
        """
        loss = newspe.spee(30, 8600, 8)
        ls = loss
        # heights_len = ls.shape[0]
        heights_len = 30
        ranges = ls.shape[1]
        fields = []
        x_axis = list(range(0, heights_len))
        for _ in range(heights_len):
            for rge in range(ranges):
                cur_ls = ls[_][rge]
                if not RadarCal.is_detected(cur_ls, Ts):
                    fields.append(rge)
                    break

        DrawUtil.draw_line_chart(x_axis, fields)

        return fields, x_axis



if __name__ == '__main__':
    rc = RadarCal()
    f = '../data/sounding_processed_hgt/stn_54511/stn_54511_2021-12-12_00UTC.npy'
    # # station = f.split('.')[-2].split('_')[-3]
    # # date = f.split('.')[-2].split('_')[-2]
    # loss, _ = rc.get_l_single(f, 9400, 8)
    # # DrawUtil.draw_l_single_heatmap(loss, title='{}-{}-{}'.format(station, date, _))
    # DrawUtil.draw_l_single_heatmap(loss, title='{}'.format(_))

    print(rc.get_Ts())
    rc.get_detected_field(None, rc.get_Ts())
    # loss = newspe.spee(30, 8600, 8)
    # DrawUtil.draw_l_single_heatmap(loss, title='30m')
