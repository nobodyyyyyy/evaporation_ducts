from cmath import exp, log, atan, sqrt

import numpy as np

from new.Util.MathUtil import MathUtil


def R_S(t: np.array, p: np.array):
    try:
        6.112 * exp(17.502 * t / (t + 240.97)) * (1.0007 + 3.46e-6 * p)
    except OverflowError as e:
        return 1
    return 6.112 * exp(17.502 * t / (t + 240.97)) * (1.0007 + 3.46e-6 * p)


def qsee(ts, P):
    es = R_S(ts, P) * 0.98
    qs = es * 0.622 / (P - 0.378 * es)
    return qs


def psiu_nps(zet):
    # fixme 我就把 zet 看成一个值了，有问题再说

    x = (1 - 15 * zet) ** .25
    psik = 2 * log((1 + x) / 2) + log((1 + x * x) / 2) - 2 * atan(x) + 2 * atan(1)
    x = (1 - 10 * zet) ** .3333
    psic = 1.5 * log((1 + x + x * x) / 3) - sqrt(3) * atan((1 + 2 * x) / sqrt(3)) + 4 * atan(1) / sqrt(3)
    f = zet * zet / (1 + zet * zet)
    psi = (1 - f) * psik + f * psic

    if zet.real > 0:
        c = min(50., (.35 * zet).real)
        psi = -((1 + 2 / 3 * zet) * 1.5 + 2 / 3 * (zet - 14.28) / exp(c) + 8.525)
        return psi

    return psi


def psit_nps(zet):
    # fixme 同上

    x = (1 - 15 * zet) ** .5
    psik = 2 * log((1 + x) / 2)
    x = (1 - 34 * zet) ** .3333
    psic = 1.5 * log((1 + x + x * x) / 3) - sqrt(3) * atan((1 + 2 * x) / sqrt(3)) + 4 * atan(1) / sqrt(3)
    f = zet * zet / (1 + zet * zet)
    psi = (1 - f) * psik + f * psic

    if zet.real > 0:
        c = min(50., (.35 * zet).real)
        psi = -((1 + 2 / 3 * zet) * 1.5 + 2 / 3 * (zet - 14.28) / exp(c) + 8.525)
        return psi

    return psi


def psiu_25(zet):
    x = (1 - 16 * zet) ** .25
    psik = 2 * log((1 + x) / 2) + log((1 + x * x) / 2) - 2 * atan(x) + 2 * atan(1)
    x = (1 - 12.87 * zet) ** .3333
    psic = 1.5 * log((1 + x + x * x) / 3) - sqrt(3) * atan((1 + 2 * x) / sqrt(3)) + 4 * atan(1) / sqrt(3)
    f = zet * zet / (1 + zet * zet)
    psi = (1 - f) * psik + f * psic

    if zet.real > 0:
        psi = -5 * zet
        return psi

    return psi


def psit_25(zet):
    x = (1 - 16 * zet) ** .5
    psik = 2 * log((1 + x) / 2)
    x = (1 - 12.87 * zet) ** .3333
    psic = 1.5 * log((1 + x + x * x) / 3) - sqrt(3) * atan((1 + 2 * x) / sqrt(3)) + 4 * atan(1) / sqrt(3)
    f = zet * zet / (1 + zet * zet)
    psi = (1 - f) * psik + f * psic
    if zet.real > 0:
        psi = -5 * zet
    return psi


def fait(zet):
    x = (1 - 16 * zet) ** 0.5
    y = (1 - 12.87 * zet) ** 0.3333
    f = zet * zet / (1 + zet * zet)
    psi = (1 - f) / x + f / y
    return psi


def kelvins2degrees(kelvins):
    return MathUtil.sub(kelvins, 273.15)


def degrees2kelvins(degrees):
    return MathUtil.add(degrees, 273.15)


def atmospheric_refractive_index_M(t, p, rh, z):
    """
    计算得到某一高度的大气折射率 M
    :param t: 温度 C
    :param p: 气压 hPa
    :param rh: 相对湿度  rh = e / E %
    :param z: 高度
    """
    E = 6.112 * exp(17.67 * t / (t + 243.5))  # 饱和水汽压【修正的 Tetens 公式】
    e = E * rh / 100
    t_k = degrees2kelvins(t)
    m = 77.6 * (p + 4810 * e / t_k) / t_k + 0.157 * z
    return m.real


def get_duct_height(m_list, z_list, caller='', debug=False):
    """
    获取波导高度前要不要先判断波导类型？
    由论文可见表面波导和蒸发波导的廓线差不多，故当作一起处理
    :param m_list: 大气折射率
    :param z_list: 高度
    :param caller:
    :return: 波导高度
    """
    if caller == '':
        caller = 'get_duct_height'
    pre = m_list[1] - m_list[0]  # todo (exception) we assume the len is always big enough
    # True: 悬空波导；False：蒸发波导或表面波导
    _flag = True if pre > 0 else False
    pre = int(pre)  # 刚开始【不能】转为整数
    z1 = m1 = -1
    # debug only
    # z1_pos = -1
    for _ in range(2, len(m_list)):
        sub = int(m_list[_] - m_list[_ - 1])
        if sub ^ pre < 0:
            if not _flag:
                return z_list[_ - 1], m_list[_ - 1]
            if z1 != -1 and m1 != -1:
                # print('[debug] get_duct_height... z1:{}, z2:{}, z1_pos:{}, z2_pos:{}'.
                #       format(z1, z_list[_ - 1], z1_pos, _ - 1))
                return z_list[_ - 1] - z1, abs(m_list[_ - 1] - m1)
            # 找第二个拐点
            z1 = z_list[_ - 1]
            m1 = m_list[_ - 1]
            # z1_pos = _ - 1
        pre = sub
    if debug:
        print('{}... cannot get duct height'.format(caller))
    return -1, -1


def cal_height(p, t, method='hypsometric'):
    """
    根据压强和气温计算高度
    :param method: 计算方法
    :param p: kPa 千帕 ！
    :param t: 摄氏度
    """
    p0 = 101.325
    if method.lower() == 'hypsometric':
        return ((pow(p0 / p, 1 / 5.257) - 1) * (t + 273.15)) / 0.0065
    elif method.lower() == 'barometric':
        return 44330 * (1 - pow(p / p0, 1 / 5.257))
    else:
        print('cal_height... method {} not supported'.format(method))
        return -1


if __name__ == '__main__':
    print(cal_height(93.0, -22.8))