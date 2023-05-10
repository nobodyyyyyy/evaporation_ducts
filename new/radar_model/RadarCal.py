import math


class RadarCal:


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