import math

import numpy as np

from new.data.DataUtil import DataUtils
from data import SP


class RadarCal:

    def __init__(self):
        self.dataset_cache = {}

    @staticmethod
    def get_Ts(f, Pt, G, D0, Bn, Ls, F0, sigma):
        # f:雷达频率(MHz)
        # Pt:雷达峰值功率(KW) [论文写的是发射功率 Pt]
        # G:天线增益(dB)
        # D0:最小信噪比(dB) D0 为雷达信号的检测因子(最小可检测信噪比)
        # Bn:接收机带宽(MHz) 与脉宽 t 满足关系：Bn ≈ 1/ t
        # Ls:系统综合损耗(dB)
        # F0:接收机噪声系数(dB)
        # sigma:目标散射截面(m*m) σ
        # 输出lossflag [门限]

        #   没用到的：Text11:目标高度(m)  Text6:发射仰角(deg)  Text5:波束宽度(deg)  Text3:天线高度(m)
        Lossflag = 135.43 + 10 * math.log10(Pt * sigma * f * f) - \
                   10 * math.log10(Bn) + \
                   2 * G - Ls - F0 - \
                   10 * math.log10(D0)
        Lossflag = Lossflag * 0.5
        return Lossflag

    @staticmethod
    def get_Ls(f, R, F):
        # Ls 电波在实际环境的单程传播损耗
        # f:雷达频率(MHz)
        # R:目标斜距(km)
        # F:传播因子(dB)
        # return Lfs - 20 * math.log10(F)
        # Ls 与大气传播环境(大气波导)密切相关,可利用电磁波传播的抛物数值方程计算得到.
        return 32.44 + 20 * math.log10(f) + 20 * math.log10(R) - 20 * math.log10(F)

    @staticmethod
    def is_detected(Ls, Ts):
        """
        Ls <= Ts 即目标能够被检测到
        """
        return Ls <= Ts

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

    def get_l_single(self, data_dir, debug=True):
        """
        拿 L_{single}
        """
        name = DataUtils.get_file_name(data_dir)
        if name in self.dataset_cache.keys():
            dataset = self.dataset_cache[name]
        else:
            dataset = np.load(data_dir, allow_pickle=True)
            self.dataset_cache[name] = dataset
        # todo 一个 data_dir 对应某一时间的文件，后续文件格式改变需要修改
        data_list = []
        for e in dataset:
            t = e['TEMP']  # 气温
            eh = e['RELH']  # 相对湿度
            p = e['PRES']  # 压强
            h = e['HGNT']  # 测量高度
            if t is None or eh is None or h is None or p is None:
                if debug:
                    print('cal_real_height... data incomplete: [{}, {}, {}, {}]'.format(t, eh, p, h))
                continue
            data_list.append([t, p, eh, h])

        ref, h = DataUtils.generate_ref_and_h(data_list)
        pre_data = RadarCal.radar_data_pre_generate(ref, h)
        loss = SP.spe(pre_data[0][0], pre_data[1][0], pre_data[2][0], pre_data[3][0], pre_data[4][0])
        loss = np.array(loss)
        return loss


if __name__ == '__main__':
    RadarCal.get_l_single()