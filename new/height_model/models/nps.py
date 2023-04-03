import sys
from cmath import sqrt, log, exp

import numpy as np

from new.Util.DuctHeightUtil import R_S, qsee, psiu_nps, psit_nps


def nps_duct_height(t, RH, ts, u, P, height=-1, stable_check=False):
    # zu = 12.5  # 风速测量高度
    # zt = 12.5  # 温度测量高度
    # zq = 12.5  # 湿度测量高度
    # zp = 12.5  # 气压测量高度

    if height != -1:
        zu = height
        zt = height
        zq = height
        zp = height

    Beta = 1.25  # 阵风系数
    von = 0.4  # Karman常数
    tdk = 273.16  # 开尔文温度转换量
    grav = 9.82  # 重力加速度
    zi = 600  # 设定近地层厚度
    us = 0 

    es = R_S(t, P) * RH / 100 
    Q = es * 0.622 / (P - 0.378 * es)  # 相对湿度转化为比湿
    Qs = qsee(ts, P) # 海表面饱和比湿
    # disp(Qs)
    
    # 空气常量

    Rgas = 287.04  # 干空气气体常数
    cpa = 1004.67  # 干空气比热
    # cpv = cpa * (1 + 0.84 * Q) 
    # rhoa = P * 100 / (Rgas * (t + tdk) * (1 + 0.61 * Q))  # 湿空气密度
    visa = 1.326e-5 * (1 + 6.542e-3 * t + 8.301e-6 * t * t - 4.84e-9 * t * t * t)  # 运动粘滞系数
    du = u - us 
    dt = ts - t - 0.00976 * zt 
    dq = Qs - Q 
    ta = t + tdk 
    # tv = ta * (1 + 0.61 * Q) * (1000 / P) ^ (Rgas / cpa)  # 虚温

    tv = (ta + 0.00976 * zt) * (1 + 0.61 * Q) 
    
    # 首次猜测
    
    ug = 0.5  
    ut = sqrt(du * du + ug * ug)  
    u10 = ut * log(10 / 1e-4) / log(zu / 1e-4)  
    usr = 0.035 * u10  
    zo10 = 0.011 * usr * usr / grav + 0.11 * visa / usr  
    Cd10 = (von / log(10 / zo10))**2
    Ch10 = 0.00115  
    Ct10 = Ch10 / sqrt(Cd10)  
    zot10 = 10 / exp(von / Ct10)

    # 减少迭代次数
    Cd = (von / log(zu / zo10))**2
    Ct = von / log(zt / zot10)  
    CC = von * Ct / Cd  
    Ribcu = -zu / zi / 0.004 / Beta**3
    Ribu = -grav * zu / ta * (dt + 0.61 * ta * dq) / ut**2

    nits = 3   # 设置迭代次数
    if Ribu.real < 0:
        zetu = CC * Ribu / (1 + Ribu / Ribcu)
    else:
        zetu = CC * Ribu * (1 + 27 / 9 * Ribu / CC)

    L10 = zu / zetu
    if zetu.real > 50:
        nits = 1

    usr = ut * von / (log(zu / zo10) - psiu_nps(zu / L10))   
    tsr = -dt * von / (log(zt / zot10) - psit_nps(zt / L10))
    qsr = -dq * von / (log(zq / zot10) - psit_nps(zq / L10))

    charn = 0.011

    if ut.real > 10:
        charn = 0.011 + (ut - 10) / (18 - 10) * (0.018 - 0.011)
    if ut.real > 18:
        charn = 0.018

    for i in range(1, nits + 1):
        zet = von * grav * zu * (tsr * (1 + 0.6078 * Q) + 0.6078 * ta * qsr) / tv / (usr * usr)
        # 严格按照NPS的文献所述的稳定度函数形式，区别位温和虚位温。
        # disp(zet)
        zo = charn * usr * usr / grav + 0.11 * visa / usr
        rr = zo * usr / visa
        L = zu / zet
        zoq = min(1.15e-4, (5.5e-5 / rr**.6).real)  # Fairall COARE3.O
        zot = zoq

        usr = ut * von / (log(zu / zo) - psiu_nps(zu / L))
        tsr = -dt * von / (log(zt / zot) - psit_nps(zt / L))
        qsr = -dq * von / (log(zq / zoq) - psit_nps(zq / L))
        Bf = -grav / tv * usr * (tsr + 0.61 * ta * qsr)
        if Bf.real > 0:
            ug = Beta * (Bf * zi)**0.333
        else:
            ug = 0.2
        ut = sqrt(du * du + ug * ug)

    if zet.real >= 0:
        _stable = True
    else:
        _stable = False

    # 计算波导高度
    ee = 0.62197  # the ratic of the gas constant for dry air to that of water vapor

    # 计算温度、比湿廓线
    h0 = zo
    h = np.arange(51)  # [0, 1, ..., 50]
    T_kuoxian = np.arange(51).astype(complex)
    Q_kuoxian = np.arange(51).astype(complex)

    T_kuoxian[0] = ts + tsr / von * (log(h0 / zot) - psit_nps(h0 / L)) - 0.00976 * h0
    Q_kuoxian[0] = Qs + qsr / von * (log(h0 / zoq) - psit_nps(h0 / L))
    for i in range(1, len(h)):
        T_kuoxian[i] = ts + tsr / von * (log(h[i] / zot) - psit_nps(h[i] / L)) - 0.00976 * h[i]
        Q_kuoxian[i] = Qs + qsr / von * (log(h[i] / zoq) - psit_nps(h[i] / L))

    P_kuoxian = np.zeros(51).astype(complex)
    P_kuoxian[0] = P*exp(2 * grav * zp / Rgas / (ta * (1 + 0.61 * Q) + (ts + tdk) * (1 + 0.61 * Qs)))

    #  首先利用测量高度上的气压计算海平面气压
    for i in range(1, len(h)):
        P_kuoxian[i] = P_kuoxian[i - 1] * exp(2 * grav * (-1) / Rgas / ((T_kuoxian[i - 1] + tdk) * (1 + 0.61 * Q_kuoxian[i - 1]) + (T_kuoxian[i] + tdk) * (1 + 0.61 * Q_kuoxian[i])))

    #  计算修正折射率廓线
    E = np.zeros(len(h)).astype(complex)
    rh = np.zeros(len(h)).astype(complex)
    M = np.zeros(len(h)).astype(complex)
    
    for i in range(0, len(h)):
        E[i] = Q_kuoxian[i] * P_kuoxian[i] / (ee + (1 - ee) * Q_kuoxian[i])
        rh[i] = 100 * E[i] / R_S(T_kuoxian[i], P_kuoxian[i])  # 相对湿度廓线
        M[i] = 77.6 * P_kuoxian[i] / (T_kuoxian[i] + tdk) - (5.6 / (T_kuoxian[i] + tdk) - 3.75e5 / (T_kuoxian[i] + tdk)**2) * E[i] + 0.1568 * h[i]
    # 修正折射率最小值对应的高度hh，即为蒸发波导高度

    _min = sys.maxsize
    idx = -1
    for _ in range(len(M)):
        if M[_] < _min:
            _min = M[_]
            idx = _
    if idx == -1:
        print('nps... error: cannot cal height')
        if stable_check:
            return 0, _stable
        return 0

    res = h[idx]
    if res > 40:
        res = 0
    if res < 0:
        res = 0

    if stable_check:
        return res, _stable
    return res


if __name__ == '__main__':
    t = 24.9
    RH = 69
    ts = 25.3
    u = 5.8
    P = 1006.7
    nps_duct_height(t, RH, ts, u, P)
