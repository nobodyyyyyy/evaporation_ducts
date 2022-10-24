﻿#!/usr/bin/env python
# -*- coding:utf-8 -*-
import time
import threading
import pandas
import math
from data import SP
# from .SP import *
import numpy as np
from cmath import exp
from scipy.interpolate import interp1d

def smooth(a,WSZ=5):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    a = np.array(a).flatten()
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))

pi = 3.1415926
def R_S(t, p):
    qa = 6.112 * np.exp(17.502 * t / (t + 241)) * (1.0007 + 3.46e-6 * p)
    return qa

def qsee(ts, P):
    es = R_S(ts, P)*0.98
    qs = es * 0.622 / (P - 0.378 * es)
    return qs

#判断悬空波导
def xuankong(array,H):
    xuankong=np.zeros(shape=8)
    MM=smooth(array)
    a=np.diff(MM)
    for i in range(3,154):
        if a[i]<0 and a[i+1]<0 and a[i+2]<0 and a[i-1]>0 and a[i-2]>0 and a[i-3]>0 and (a[i]+a[i+1]+a[i+2])<-3:
            for ii in range(i+1,154):
                if a[ii]>0 and MM[ii]>MM[0]:
                    xuankong[0]=1#表示发生了悬空波导
                    xuankong[1]=MM[0]#波导最低处的M值
                    xuankong[2]=MM[ii]#波导顶高度处的M值
                    xuankong[3]=MM[i]#波导底高度出的M值
                    xuankong[4]=H[ii]#波导顶高度
                    xuankong[5]=H[i]#波导底高度
                    xuankong[6]=abs(xuankong[3]-xuankong[2])#波导强度
                    xuankong[7]=abs(xuankong[5]-xuankong[4])#波导厚度
                    break
            break
    xuankong=xuankong.reshape((-1,1))
    return xuankong

#判断表面波导
def biaomian(array,H):
    biaomian=np.zeros(shape=5)
    MM=smooth(array)
    a=np.diff(MM)
    for i in range(0,16):
        if a[0]<-3:
            for ii in range(1,17):
                if a[ii]>0:
                    biaomian[0] = 1#表示发生了表面波导
                    biaomian[1]=MM[0]#波导最低处的M值
                    biaomian[2]=MM[ii]#波导顶高度出的M值
                    biaomian[3]=H[ii]#波导顶高度，对于表面波导，波导顶高度即为波导厚度
                    biaomian[4]=abs(biaomian[2]-biaomian[1])#波导强度
                    break
            break
    biaomian=biaomian.reshape((-1,1))
    return biaomian

#p气压 t温度，shidu相对湿度，z高度,计算得到某一高度的大气折射率
def zheshelv(t,p,shidu,z):
    #e为饱和水汽压
    e=6.112*exp(17.67*t/(t+243.5))
    e=e*shidu/100
    T = t+273.15
    M=77.6*(p+4810*e/T)/T+157*z
    return M.real

#输入data，生成其他算法需要的廓线和高度数据
# axis = [Temp, Press, Hum, H]
def generate_data(data, axis=[1, 2, 3, 0]):
    h = np.zeros((data.shape[0], 1))
    ref = np.zeros((data.shape[0], 1))
    for i in range(0, data.shape[0]):
        h[i][0] = data[i][axis[3]]
        ref[i][0] = zheshelv(data[i][axis[0]], data[i][axis[1]], data[i][axis[2]], data[i][axis[3]])
    return ref, h

#输出蒸发波导高度
def zhengfaM(ref, h):
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
        for j in range(ref.shape[0]-1):
            if r[j, i] <= 0 and r[j+1, i] > 0 and r[0, i] < 0:
                hh1[n, i] = h[j, i]
                g1[n, i] = h[j, i]
                ddm1[n, i] = ref[0, i]-ref[j, i]
                tidu1[n, i] = ddm1[n, i]/hh1[n, i]
                if g1[n, i] < 40:
                    flag1[n, i] = 1
                else:
                    flag1[n, i] = 2
                n += 1
            elif r[j+1, i] <= 0 and r[j, i] > 0 and r[0, i] > 0:
                delm2[k, i] = ref[j, i]
                for ss in range(ref.shape[0]-j-1):
                    if r[j+ss, i] <=0 and  r[j+ss+i, i] > 0:
                        if ref[j+ss, i] > ref[0, i]:
                            flag2[k, i] = 3
                            g2[k, i] = h[j, i]
                            ddm2[k, i] = ref[j+ss, i]-delm2[k ,i]
                            for b in range(j-1):
                                if np.abs(h[b, i]- h[j, i]) < 0.1:
                                    hh2[k, i] = h[j+ss, i] - g2[k, i]
                                    tidu2[k, i] = ddm2[k, i]/hh2[k, i]
                    elif ref[j+ss, i] < ref [0, i]:
                        flag2[k, i] = 4
                        ddm2[k, i] = ref[j+ss, i] - delm2[k, i]
                        hh2[k, i] = h[j+ss, i] - h[j, i]
                        tidu2[k, i] = delm2[k, i]/g2[k, i]
                cc1[k, i] =delm2[k, i]/g2[k, i]
                k += 1
    res = np.zeros((2, ref.shape[1]))
    count = 0
    for i in range(0, flag1.shape[1]):
        for j in range(0, flag1.shape[0]):
            if (flag1[j][i] == 1):
                # 判定发生了蒸发波导，输出蒸发波导高度
                for k in range(0, g1.shape[0]):
                    if (g1[k][i] != 0):
                        res[0][count]=i+1
                        res[1][count]=g1[k][i]
                        count=count+1
                        break
                break
    if(count!=0):
        return res[1]
    else:
        return []

# data的传入模式为气温、海表温、湿度、风速、压强
# T：气温
# TS：海温
# eh：相对湿度
# U：风速
# p：压强
def zhengfaMs(t,ts,RH,u,P):
    zu = 12
    # 温度测量高度
    zt = 12
    # 湿度测量高度
    zq = 12

    zz = 12
    # 阵风系数
    Beta = 1.25
    # Karman常数
    von = 0.4
    # 开尔文温度转换量
    tdk = 273.16
    # 重力加速度
    grav = 9.8
    # 设定近地层厚度
    zi = 600
    us = 0
    # 普朗特常数
    R = 1
    # the ratic of the gas constant for dry air to that of water vapor
    ee = 0.62197
    zs = -1
    # 粗糙长度
    z0m = 0.0002
    # 温湿粗糙长度
    z0h = 0.0001
    # 湿度粗糙长度
    z0q = z0h
    e1 = R_S(t, P) * RH / 100
    # 相对湿度转化为比湿
    Q = e1 * 0.622 / (P - 0.378 * e1)
    # 海表面饱和比湿
    Qs = qsee(ts, P)
    # *(1000 / P) ** (0.286) * (1 + 0.608 * Q);
    theta = (t + tdk)
    # *(1000 / P) ** (0.286) * (1 + 0.608 * Qs)
    thetas = (ts + tdk)
    # 蒸发潜热
    Le = (2.501 - 0.00237 * (ts + tdk)) * 10 ** 6
    # 干空气气体常数
    Rgas = 287.1
    # 干空气比热
    cpa = 1004.67
    # cpv = cpa * (1 + 0.84 * Q);
    # 湿空气密度
    rhoa = P * 100 / (Rgas * (t + tdk) * (1 + 0.61 * Q))
    # rhoa = 1.12; % dry空气密度
    # 运动粘滞系数
    visa = 1.326e-5 * (1 + 6.542e-3 * t + 8.301e-6 * t * t - 4.84e-9 * t * t * t)
    e = Q * P / (ee + (1 - ee) * Q)
    du = u - us
    # 改成位温
    dt = ts - t - 0.0098 * zt
    # print("Qs ", Qs)
    # print("Q ", Q)
    dq = Qs - Q
    ta = t + tdk
    # 通过Rib判断稳定性
    Rib = -grav * zu / ta * (dt + 0.61 * ta * dq) / u ** 2
    A = -0.01 * rhoa * grav * (77.6 / ta + 77.6 * 4810 * Q / (ta * ta * (e + (1 - ee) * Q))) - grav * (
            P - (1 - ee) * e) / cpa * (
                -77.6 / ta / ta - 2 * 4810 * 77.6 * Q / (ta * ta * ta * (ee + (1 - ee) * Q)))
    B = (P / 1000) ** (Rgas / cpa) * (
            -77.6 * P / (ta * ta) - 2 * 77.6 * 4810 * P * Q / (ta * ta * ta * (ee + (1 - ee) * Q)))
    C = 4810 * 77.6 * P * ee / (ta * ta * (ee + (1 - ee) * Q) ** 2)
    # ZL: stability parameter
    # Rib: bulk Richardson number
    # zz: observation height
    # z0m: aerodynamic roughness length
    # z0h: thermal roughness length
    # function[CM, CH, ZL] = Li14_main(Rib, zz, z0m, z0h, zs)
    Lom = np.log(zz / z0m)
    Loh = np.log(zz / z0h)
    zkbm = np.log(z0m / z0h)
    zzz0m = zz / z0m
    cmum = 2.59
    cmuh = 0.95
    cnu = 0.5
    clam = 1.5
    caa = 6.1
    cbb = 2.5
    ccc = 5.3
    cdd = 1.1
    cgammam = 16.0
    cgammah = 16.0
    calpham = 0.25
    calphah = 0.5
    p2 = pi / 2
    zzz0h = zzz0m * np.exp(zkbm)
    zL0M = np.log(zzz0m)
    zL0H = zL0M + zkbm
    # 为什么要乘0.06
    zzzs = zzz0m * 0.06
    # exp里面写错了吧？
    zfacM = np.log(1. + clam / cmum / zzzs) * np.exp(-cmum * zzzs) / clam
    zfacM2 = (1. + cnu / cmum / zzzs)
    zfacH = np.log(1. + clam / cmuh / zzzs) * np.exp(-cmuh * zzzs) / clam
    zfacH2 = (1. + cnu / cmuh / zzzs)
    bet1 = np.array([1, np.log(Lom), np.log(Lom) * np.log(Lom), (Loh - Lom), np.log(Lom) * (Loh - Lom),
                     np.log(Lom) * np.log(Lom) * (Loh - Lom), (Loh - Lom) * (Loh - Lom),
                     (Loh - Lom) * (Loh - Lom) * np.log(Lom)])
    bet2 = np.array(
        [1, Rib, Rib ** 2, Rib ** 3, (Loh - Lom), Rib * (Loh - Lom), Rib ** 2 * (Loh - Lom), Rib ** 3 * (Loh - Lom),
         (Loh - Lom) ** 2, Rib * (Loh - Lom) ** 2, Rib ** 2 * (Loh - Lom) ** 2, (Loh - Lom) ** 3,
         Rib * (Loh - Lom) ** 3,
         Lom, Rib * Lom, Rib ** 2 * Lom, Rib ** 3 * Lom, Lom * (Loh - Lom), Rib * Lom * (Loh - Lom),
         Rib ** 2 * Lom * (Loh - Lom), Lom * (Loh - Lom) ** 2, Rib * Lom * (Loh - Lom) ** 2, Lom * (Loh - Lom) ** 3,
         Lom ** 2,
         Rib * Lom ** 2, Rib ** 2 * Lom ** 2, Lom ** 2 * (Loh - Lom), Rib * Lom ** 2 * (Loh - Lom),
         Lom ** 2 * (Loh - Lom) ** 2, Lom ** 3, Rib * Lom ** 3, Lom ** 3 * (Loh - Lom)])
    bet3 = np.array(
        [Lom ** -3 * Loh ** -1, Lom ** -1 * Loh ** -3, (-Rib / (1 - Rib)) * Loh ** -3, (-Rib / (1 - Rib)) * Lom ** -3,
         Lom ** -2 * Loh ** -2, (-Rib / (1 - Rib)) * Lom ** -1 * Loh ** -2, (-Rib / (1 - Rib)) * Lom ** -2 * Loh ** -1,
         Loh ** -3, Lom ** -3,
         Lom ** -1 * Loh ** -2, Lom ** -2 * Loh ** -1, (-Rib / (1 - Rib)) * Loh ** -2, (-Rib / (1 - Rib)) * Lom ** -2,
         (-Rib / (1 - Rib)) * Lom ** -1 * Loh ** -1, Loh ** -2, Lom ** -2, Lom ** -1 * Loh ** -1,
         (-Rib / (1 - Rib)) * Loh ** -1,
         (-Rib / (1 - Rib)) * Lom ** -1, Loh ** -1, Lom ** -1, (-Rib / (1 - Rib)), 1])
    if np.isnan(Rib) or (Rib == 0) or (Rib <= -5):
        zs = 1
        CM = np.nan
        CH = np.nan
        ZL = np.nan
    if zs < 0:
        # for stable conditions
        if Rib > 0:
            # select region
            if zz / z0m <= 160 and zz / z0m >= 10 and z0m / z0h >= 0.607 and z0m / z0h <= 100:
                # print(region 1")
                coefficients1 = np.array([[0.3095, -0.2852, 0.07955, 0.03388, -0.01605, 0, 0, -1.079 * 10 ** -4],
                                          [0.3219, -0.2613, 0.06753, 0.04838, -0.03101, 0.003908, -0.001780, 0.001165],
                                          [0.3545, -0.2569, 0.06609, 0.05837, -0.03934, 0.005643, -0.003381, 0.002194],
                                          [0.4390, -0.3133, 0.08619, 0.08930, -0.07112, 0.014030, -0.005965, 0.003806],
                                          [0.6887, -0.5375, 0.16160, 0.17540, -0.15640, 0.034890, -0.012770, 0.008101],
                                          [1.7060, -1.6200, 0.52310, 0.51240, -0.50260, 0.123900, -0.035770, 0.02238]])
                coefficients2 = np.array(
                    [[-1.134, 31.1, -71.16, 227.4, -0.2094, 3.293, -20.11, 14.42, 0.1476, -0.07325, 0.5627, -0.01178,
                      0.0218, 1.405, -32.47, 46.59, -38.25, -0.2286, -1.097, -0.3394, 0, 0, 0, 0, 10.71, 0, 0, 0,
                      0, -0.007485, -0.9671, 0.003402],
                     [0, 86.35, 0, 0, -11.53, 194.9, -975.4, 1472, -2.535, 28.24, -61.13, -0.2378, 0.7405, 13.6, -316.2,
                      1067, -1494, 8.023, -91.31, 213.7, 1.035, -5.072, 0.03622, -4.699, 97.46, -152.4, -1.704, 9.069,
                      -0.09576, 0.4446, -7.991, 0.1138],
                     [0, -280.4, 3235, -6165, -10.64, 193.8, -1194, 2161, -4.603, 52.02, -110.7, -0.5367, 1.503, 30.26,
                      -314.9, 186, 0, 9.038, -87.06, 198.6, 1.529, -7.439, 0.07369, -10.71, 122.1, -76.91, -2.035,
                      8.248,
                      -0.1263, 1.015, -10.96, 0.1426],
                     [0, 0, 0, 0, 0, 0, -12.37, 0, 0, 11.99, -15.63, -0.3157, 0.2948, 0, 0, -108.1, 317.8, 0, -12.52, 0,
                      0, -1.025, 0.04669, -1.896, 28.39, -14.19, 0, 2.214, -0.01472, 0.3069, -3.635, -0.008769],
                     [0, 0, 0, 0, 0, 1.113, -97.56, 159.4, 0, 16.33, -25.67, -0.6447, 0.9718, 6.821, -57.13, 227.3,
                      -244,
                      0.9287, -17.88, 34.41, 0.319, -2.452, 0.08583, -2.195, 22.21, -31.44, -0.1355, 1.976, -0.04636,
                      0.1708,
                      -1.623, 0],
                     [0, -17.32, 8.773, 0, 0, 0, 0, 0, 1.919, 0, 0.2679, -0.2892, 0, 10.27, 0, 0, 0, -3.457, -1.617, 0,
                      -0.07536, 0, 0.05146, -3.108, 7.948, -2.985, 0.8751, 0.3139, -0.05131, 0.2598, -0.8513, -0.05427],
                     [0, -6.343, 7.66, -0.7661, 0.0125, -2.203, 0.8896, -0.1273, -0.00827, 0.3327, -0.04613, 0,
                      -0.04968, 7.513,
                      0, -4.799, 0.5598, -1.612, 0, 0, 0.4666, 0.0605, -0.01808, 0, 2.442, 0.1584, 0, -0.04377, -0.0694,
                      -0.1675,
                      -0.2181, 0.05052]])
                Ribc1 = sum(bet1 * coefficients1[0, :])
                Ribc2 = sum(bet1 * coefficients1[1, :])
                Ribc3 = sum(bet1 * coefficients1[2, :])
                Ribc4 = sum(bet1 * coefficients1[3, :])
                Ribc5 = sum(bet1 * coefficients1[4, :])
                Ribc6 = sum(bet1 * coefficients1[5, :])
                if Rib > 0 and Rib <= Ribc1:
                    ZL = Rib * sum(bet2 * coefficients2[0, :])
                elif Rib > Ribc1 and Rib <= Ribc2:
                    ZL = Rib * sum(bet2 * coefficients2[1, :])
                elif Rib > Ribc2 and Rib <= Ribc3:
                    ZL = Rib * sum(bet2 * coefficients2[2, :])
                elif Rib > Ribc3 and Rib <= Ribc4:
                    ZL = Rib * sum(bet2 * coefficients2[3, :])
                elif Rib > Ribc4 and Rib <= Ribc5:
                    ZL = Rib * sum(bet2 * coefficients2[4, :])
                elif Rib > Ribc5 and Rib <= Ribc6:
                    ZL = Rib * sum(bet2 * coefficients2[5, :])
                elif Rib > Ribc6:
                    ZL = Rib * sum(bet2 * coefficients2[6, :])
            elif zz / z0m <= 10 ** 5 and zz / z0m > 160 and z0m / z0h >= 0.607 and z0m / z0h <= 100:
                # print(region 2")
                coefficients1 = np.array([[0, 0.08606, -0.03048, 0.09019, -0.07682, 0.01693, 0, 0],
                                          [0.2002, 0, -0.01589, 0, 0.00367, 0, 0.005057, -0.002399],
                                          [0.4499, 0, -0.02397, 0.0388, -0.01145, 0, 0, 0]])
                coefficients2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9996, 0, 56.57, 0, -0.1456, 0,
                                           -12.1, 0, 0.1303, 0, 0, 0.295, 0, 0.005508, -0.0359,
                                           4.067 * 10 ** -4 * 10 ** -4, 0, 0, 0],
                                          [0, 0, 0, 0, 0, -12.35, 0, 0, 0, 0.5183, 0, 0, 0, 0.8247, 0, 112.5, 0,
                                           -0.09054, 0, -2.249, 0.01653, 0, 0, 0, 0.8326, -9.554, 0, 0.07022, -0.001333,
                                           0, 0, 0],
                                          [0, 41.53, 0, 0, -1.616, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15.82, -27.37, 0, 0, 0, 0,
                                           0, 0.02288, 0, 0.1062, -0.9992, 1.56, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, -2.57, -2.91, 0, 0, 0.874, 0.3377, 0, -0.002092, -0.01343, 7.453,
                                           5.4, -1.623, 0.1999, 0, 0.4753, 0, -0.2047, -0.02581, 0, -0.9043, -0.3386,
                                           0.04556, 0.04682, -0.01924, 0.01217, 0.03944, 0.006516, -0.003571]])

                Ribc1 = sum(bet1 * coefficients1[0, :])
                Ribc2 = sum(bet1 * coefficients1[1, :])
                Ribc3 = sum(bet1 * coefficients1[2, :])
                if Rib > 0 and Rib <= Ribc1:
                    ZL = Rib * sum(bet2 * coefficients2[0, :])
                elif Rib > Ribc1 and Rib <= Ribc2:
                    ZL = Rib * sum(bet2 * coefficients2[1, :])
                elif Rib > Ribc2 and Rib <= Ribc3:
                    ZL = Rib * sum(bet2 * coefficients2[2, :])
                elif Rib > Ribc3:
                    ZL = Rib * sum(bet2 * coefficients2[3, :])
            elif zz / z0m <= 80 and zz / z0m >= 10 and z0m / z0h > 100 and z0m / z0h <= 10 ** 7:
                # print(region 3")
                coefficients1 = np.array([[0.3063, -0.2849, 0.07886, 0.03104, -0.01423, -5.632 * 10 ** -4,
                                           3.684 * 10 ** -6, -2.926 * 10 ** -6],
                                          [0.3555, -0.3002, 0.07855, 0.02617, -0.004769, -0.004012, -1.298 * 10 ** -5,
                                           9.907 * 10 ** -6],
                                          [0.5064, -0.4282, 0.1229, 0.02138, 0, -0.00441, 0, 0],
                                          [1.638, -1.743, 0.5813, 0.04471, -0.01874, 0, 0, 0]])
                coefficients2 = np.array([[2.001, -0.7876, 0, 60.42, -0.1401, -0.1085, -2.065, -2.98, 0.01334, 0.0213,
                                           0.1963, -3.704 * 10 ** -4, -0.002957, -1.442, 1.047, 0, 0, 0, 0, -1.121, 0,
                                           0.0273, 0, 0.6868, 0, 3.82, -0.01898, -0.1228, 2.845 * 10 ** -4, -0.06543,
                                           0.1469, 0.00179]
                                          [
                                              0, 0, 0, 368.9, 3.514, -8.524, -18.05, -4.852, 0.08174, 0.5791, 0.1207, -0.007021, 0, 1.207, -31.68, 32.78, -25.65, -2.096, 2.222, 0.3871, -0.004486, -0.06669, 0.001086, -0.07632, 14.32, 2.353, 0.3396, -0.3281, -3.6 * 10 ** -4, 0, -1.505, -0.01529]
                                          [
                                              -68.85, 756.9, -1100, 0, 0, -30.13, 86.99, 5.71, 0.7274, -2.554, -0.2169, 0.01587, 0.003912, 76.25, -874.1, 1636, -1040, 4.942, -17.32, 14.97, -0.09096, 0.2281, -0.002971, -21.66, 232.4, -224.1, -1.724, 3.144, -4.477 * 10 ** -4, 1.875, -18.02, 0.1523]
                                          [
                                              -1.514, 0, 0, 19.63, 0.559, 0, 0, -2.424, -0.002248, 0, 0.1259, 8.267 * 10 ** -4, -0.004141, -8.751, 51.96, -76.51, 27.69, -1.349, 1.297, -0.09621, 0, 0, 2.192 * 10 ** -4, 3.734, -6.438, 6.284, 0.2422, -0.2272, 0, -0.4111, 0.2556, -0.009961]
                                          [
                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.413 * 10 ** -4, 7.107 * 10 ** -5, 0, 1.905, -1.761, 0.3658, -0.05227, 0, 0, 0, 0, 0, 2.165, 0.6139, -0.1166, -0.07307, 0.005656, 0, -0.3134, 0, 0.008105]])

                Ribc1 = sum(bet1 * coefficients1[0, :])
                Ribc2 = sum(bet1 * coefficients1[1, :])
                Ribc3 = sum(bet1 * coefficients1[2, :])
                Ribc4 = sum(bet1 * coefficients1[3, :])

                if Rib > 0 and Rib <= Ribc1:
                    ZL = Rib * sum(bet2 * coefficients2[0, :])
                elif Rib > Ribc1 and Rib <= Ribc2:
                    ZL = Rib * sum(bet2 * coefficients2[1, :])
                elif Rib > Ribc2 and Rib <= Ribc3:
                    ZL = Rib * sum(bet2 * coefficients2[2, :])
                elif Rib > Ribc3 and Rib <= Ribc4:
                    ZL = Rib * sum(bet2 * coefficients2[3, :])
                elif Rib > Ribc4:
                    ZL = Rib * sum(bet2 * coefficients2[4, :])

            elif zz / z0m <= 10 ** 5 and zz / z0m > 80 and z0m / z0h > 100 and z0m / z0h <= 10 ** 7:
                # print(region 4")
                coefficients1 = np.array([[0.09742, 0, -0.01096, 0.04544, -0.03299, 0.006383, 0, 0],
                                          [0.17680, 0, -0.01434, 0.03558, -0.02059, 0.003327, 0, 0],
                                          [0.36360, 0, -0.02240, 0.04607, -0.02506, 0.004152, 0, 0]])
                coefficients2 = np.array([[0, 0, 0, 0, 0, 0, -6.267, 0, 0, 0.09808, 0, 0, 0, 0.5961, 0, 18.49, 34.53,
                                           -0.0845, -0.5106, -0.3543, 0.004555, 0, -9.402 * 10 ** -5, 0.05628, 0.8075,
                                           0, 0, 0.01631, -3.8 * 10 ** -5, -0.00189, -0.03755, 5.177 * 10 ** -5],
                                          [-3.528, 0, 0, 0, -0.2511, 0, -10.06, 0, 0, 0.1809, 0, 0, 0, 1.375, 2.951,
                                           68.09, 0, 0, -1.361, 0, 0.003711, 0, 0, -0.02359, 0.305, -3.765, -0.001535,
                                           0.07098, -2.577 * 10 ** -4, 0, 0, 0],
                                          [0, 0, 0, 0, -1.018, 0, 0, 0, 0, 0, 0, 6.74 * 10 ** -5, 0.001341, -2.404,
                                           41.12, -48.05, 24.94, -0.06671, 0, -0.1319, 0.006818, 0, -1.788 * 10 ** -4,
                                           0.5172, -4.023, 2.074, 0, 0, 0, -0.0192, 0.125, 0],
                                          [0, 0, -8.306, 1.212, 0, 0, 0, 0, 0, 0.0279, 0, 6.853 * 10 ** -4,
                                           -9.314 * 10 ** -4, 5.253, 7.626, -0.2889, 0.06073, -0.3959, -0.07098,
                                           0.003821, 0, 0, 0, -0.5006, -0.7376, 0, 0.04853, 0.002956, 0, 0.01968, 0.025,
                                           -0.001897]])

                Ribc1 = sum(bet1 * coefficients1[0, :])
                Ribc2 = sum(bet1 * coefficients1[1, :])
                Ribc3 = sum(bet1 * coefficients1[2, :])

                if Rib > 0 and Rib <= Ribc1:
                    ZL = Rib * sum(bet2 * coefficients2[0, :])
                elif Rib > Ribc1 and Rib <= Ribc2:
                    ZL = Rib * sum(bet2 * coefficients2[1, :])
                elif Rib > Ribc2 and Rib <= Ribc3:
                    ZL = Rib * sum(bet2 * coefficients2[2, :])
                elif Rib > Ribc3:
                    ZL = Rib * sum(bet2 * coefficients2[3, :])

            elif zz / z0m <= 40 and zz / z0m >= 10 and z0m / z0h > 10 ** 7 and z0m / z0h <= 10 ** 11:
                # print(region 5")
                coefficients1 = np.array([[0, 0, 0, 0.04825, -0.01677, -0.004762, -5.212 * 10 ** -4, 2.768 * 10 ** -4],
                                          [0, 0, 0.08807, 0.05219, -0.01822, -0.01245, -8.5 * 10 ** -4,
                                           7.516 * 10 ** -4],
                                          [0, 0, 0.1219, 0.0583, -0.02373, -0.01224, -0.001081, 9.539 * 10 ** -4],
                                          [0, 0, 0.1609, 0.07789, -0.04617, -0.00736, -0.001399, 0.001238],
                                          [0.4437, 0, 0, 0.1349, -0.1388, 0.03347, -0.00119, 0.001095]])
                coefficients2 = np.array([[0, 0, -2.541, 25.22, -0.03201, 0.1159, -0.5745, -0.8502, 0.00208, -0.001668,
                                           0.03737, -1.828 * 10 ** -5, -3.967 * 10 ** -4, 0.4298, -0.03339, 0.05692, 0,
                                           -0.0233, 0, -0.3158, 0, 0.007595, 0, 0, 0, 1.793, 0.00249, -0.05666, 0, 0,
                                           0.129, 0],
                                          [0, 77.11, -201.2, 386.1, -0.6831, 0, -7.571, -8.978, 0.07136, 0, 0.3442, 0,
                                           -0.003421, 0, -31.72, 2.558, 0, 0, 2.695, -2.449, -0.05044, 0.05465,
                                           -6.869 * 10 ** -5, 0.3612, 0, 18.63, 0.1236, -0.837, 0.008316, -0.06987,
                                           0.8756, -0.01959],
                                          [-207.7, 880, -1550, 2201, 0, 11.61, -96.51, 0, 0.5093, 0.8873, 0.2868,
                                           -0.001909, -0.004313, 189.4, -543.8, 324, -80.25, -5.403, 14.95, -1.706,
                                           -0.4221, 0.164, -0.00111, -53.83, 89.42, 34.6, 2.704, -4.573, 0.0718, 4.95,
                                           -3.112, -0.3287],
                                          [-587.1, 2726, -3759, 1605, -9.376, -4.513, 70.55, -58.16, 0.1711, -0.9373,
                                           1.132, -0.006865, -0.001126, 286.9, -903.7, 407.6, 260.2, 0, 14.82, -26.07,
                                           0.01062, 0.2099, 9.863 * 10 ** -4, -44.24, 98.98, 22.67, -0.01096, -1.67,
                                           -0.01056, 2.138, -4.604, 0.054],
                                          [0, 7.886, -0.5889, 0, -0.4057, 0, -0.5218, 0, 0.01745, -0.01349, 0.01468, 0,
                                           0, 0, 0, 0, 0, 0, 0.2908, 0.1992, -0.003177, -0.00933, 0, 0.7321, 2.304,
                                           -2.456, -0.09448, 0.007636, 0.002124, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.08919, 0, 0, 0, 0, 0, 0,
                                           2.053, 0.2534, -0.2585, -0.0338, 0.004269, 0, -0.3116, 0.1241, 0]])

                Ribc1 = sum(bet1 * coefficients1[0, :])
                Ribc2 = sum(bet1 * coefficients1[1, :])
                Ribc3 = sum(bet1 * coefficients1[2, :])
                Ribc4 = sum(bet1 * coefficients1[3, :])
                Ribc5 = sum(bet1 * coefficients1[4, :])

                if Rib > 0 and Rib <= Ribc1:
                    ZL = Rib * sum(bet2 * coefficients2[0, :])
                elif Rib > Ribc1 and Rib <= Ribc2:
                    ZL = Rib * sum(bet2 * coefficients2[1, :])
                elif Rib > Ribc2 and Rib <= Ribc3:
                    ZL = Rib * sum(bet2 * coefficients2[2, :])
                elif Rib > Ribc3 and Rib <= Ribc4:
                    ZL = Rib * sum(bet2 * coefficients2[3, :])
                elif Rib > Ribc4 and Rib <= Ribc5:
                    ZL = Rib * sum(bet2 * coefficients2[4, :])
                elif Rib > Ribc5:
                    ZL = Rib * sum(bet2 * coefficients2[5, :])

            elif zz / z0m <= 10 ** 5 and zz / z0m > 40 and z0m / z0h > 10 ** 7 and z0m / z0h <= 10 ** 11:
                # print(region 6")
                coefficients1 = np.array([[0, 0, 0, 0.05594, -0.03245, 0.005037, -3.654 * 10 ** -4, 1.135 * 10 ** -4],
                                          [0.1945, 0, 0, 0.03347, -0.02116, 0.002301, 0, 8.92 * 10 ** -5],
                                          [0.4288, -0.1436, 0.01635, 0.03207, -0.01382, 0.001571, 1.326 * 10 ** -5,
                                           -6.424 * 10 ** -6]])
                coefficients2 = np.array([[0, -7.864, 0, 0, -0.02699, 0.7414, -1.114, 0, 0, 0, 0, 0, 1.281 * 10 ** -4,
                                           0.244, 1.743, 4.749, 11.28, 0, -0.3093, -0.2208, 0, 0.003674, 0, 0.04168,
                                           0.4341, 0.6518, -0.00208, 0, 2.895 * 10 ** -5, 0, -0.01307,
                                           1.425 * 10 ** -5],
                                          [0.4383, 0, 0, 0, 0, -4.81, 5.094, -1.159, 0.04547, 0, -0.1233,
                                           -5.595 * 10 ** -4, 0.002459, 0, 0, 44.44, 0, 0, 0, -0.6068, -0.005459, 0, 0,
                                           0, 0.9983, -2.874, -0.00152, 0.01501, 3.541 * 10 ** -4, 0.006587, -0.04253,
                                           -3.659 * 10 ** -4],
                                          [0, -41.74, 177, -118.2, 0, -4.006, -0.5102, 0, 0, 0.0567, 0.1868, 0.002457,
                                           -0.006455, 0, 27.45, -17.37, -7.74, 0, 0, 0.0117, -0.01576, 0.02102,
                                           -1.975 * 10 ** -5, -0.1563, -2.085, 0.3443, 0.03278, -0.0325,
                                           5.167 * 10 ** -4, 0.008163, 0.0854, -0.001602],
                                          [-6.744, 8.8, -13.03, 2.203, -0.1139, -0.06103, 0.2406, -0.04635, 0.01341,
                                           -0.002749, 5.316 * 10 ** -6, -1.434 * 10 ** -4, 0, 6.511, 6.369, -0.175,
                                           0.03419, -0.3147, -0.06781, -2.026 * 10 ** -4, 0.002444, 2.616 * 10 ** -4,
                                           -5.149 * 10 ** -6, -0.6219, -0.598, 0.002868, 0.03359, 0.003178,
                                           -1.423 * 10 ** -4, 0.02407, 0.0188, -0.001167]])

                Ribc1 = sum(bet1 * coefficients1[0, :])
                Ribc2 = sum(bet1 * coefficients1[1, :])
                Ribc3 = sum(bet1 * coefficients1[2, :])

                if Rib > 0 and Rib <= Ribc1:
                    ZL = Rib * sum(bet2 * coefficients2[0, :])
                elif Rib > Ribc1 and Rib <= Ribc2:
                    ZL = Rib * sum(bet2 * coefficients2[1, :])
                elif Rib > Ribc2 and Rib <= Ribc3:
                    ZL = Rib * sum(bet2 * coefficients2[2, :])
                elif Rib > Ribc3:
                    ZL = Rib * sum(bet2 * coefficients2[3, :])

            elif zz / z0m <= 40 and zz / z0m >= 10 and z0m / z0h > 10 ** 11 and z0m / z0h <= 1.07 * 10 ** 13:
                # #print(region 7")
                coefficients1 = np.array([[0, 0, 0, 0.03681, -0.007664, -0.005619, -1.211 * 10 ** -4, 0],
                                          [0, 0, 0, 0.03655, 0, -0.009977, -2.691 * 10 ** -4, 1.057 * 10 ** -4],
                                          [0, 0, 0, 0.03822, 0, -0.01036, -3.658 * 10 ** -4, 1.769 * 10 ** -4],
                                          [0, 0, 0, 0.0384, 0, -0.009243, -3.629 * 10 ** -4, 1.471 * 10 ** -4],
                                          [0, 0, 0, 0.05616, -0.02275, 0, -5.172 * 10 ** -4, 2.261 * 10 ** -4],
                                          [0, 0, 0, 0.1472, -0.1144, 0.02796, -0.001218, 5.835 * 10 ** -4]])
                coefficients2 = np.array([[-1.412, 6.658, -5.68, 11.9, 0.1285, -0.111, -0.2095, -0.3181, -0.004693,
                                           0.004467, 0.01324, 6.64 * 10 ** -5, -2.023 * 10 ** -4, 0.7122, -4.599, 2.705,
                                           0, -0.04962, 0.01147, -0.1621, 0.001459, 0.003514, -2.01 * 10 ** -5,
                                           0.003692, 1.299, 0.6516, 0, -0.03414, 2.84 * 10 ** -5, 6.293 * 10 ** -4,
                                           -0.02559, 0],
                                          [-4.502, 40.44, 37.42, 0, 0.3067, -5.444, 2.053, 0, 0.05302, 0, -0.01586, 0,
                                           0, 1.663, -28.1, -11.02, 0, 0.1172, 1.979, -0.7285, -0.0293, 0.01334, 0,
                                           -0.4475, 5.193, 5.593, -0.009728, -0.3375, 0.00347, 0, 0, 0],
                                          [-104.2, 136.3, 233.3, 0, 13.8, -37.21, 10.33, 0, -0.1157, 0.5542, -0.2568, 0,
                                           0, 16.56, 0, -114.4, 0, -3.238, 7.578, 0, 0, -0.06568, 0, 0, -0.1495, 18.12,
                                           0.167, -0.6387, 0.00428, 0, 0, 0],
                                          [542.4, -1845, 2157, 0, -3.691, 3.33, -45.62, 0, 0.1434, 0.4557, 0.08936, 0,
                                           0, -263.7, 677.7, -644.2, 0, 4.44, -3.037, 10.93, -0.08875, -0.1436, 0,
                                           32.93, -56.58, 53.14, -0.951, 0, 0.02119, 0, 0, 0],
                                          [178.4, 158.8, -480.9, 0, -31.49, 47.56, -4.153, 0, 0.3998, -0.8692, 0.2504,
                                           0, 0, -37.94, -147.8, 144.7, 0, 9.904, -7.914, -2.224, -0.1235, 0.1631, 0, 0,
                                           23.51, -2.645, -0.7278, -0.1801, 0.008599, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20.56, -13.42, 3.002, -0.5254, 0,
                                           0, 0, 0, 2.282 * 10 ** -4, 0, -2.349, 0.628, 0.2176, 0.02067, -0.005396,
                                           -0.4148, 0.02245, 0.01163],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.06758, 0, 0.003671, 0,
                                           -6.967 * 10 ** -4, 0, 0, 0.6983, -0.1455, 0, 0, -4.282 * 10 ** -4, 0, 0, 0]])

                Ribc1 = sum(bet1 * coefficients1[0, :])
                Ribc2 = sum(bet1 * coefficients1[1, :])
                Ribc3 = sum(bet1 * coefficients1[2, :])
                Ribc4 = sum(bet1 * coefficients1[3, :])
                Ribc5 = sum(bet1 * coefficients1[4, :])
                Ribc6 = sum(bet1 * coefficients1[5, :])

                if Rib > 0 and Rib <= Ribc1:
                    ZL = Rib * sum(bet2 * coefficients2[0, :])
                elif Rib > Ribc1 and Rib <= Ribc2:
                    ZL = Rib * sum(bet2 * coefficients2[1, :])
                elif Rib > Ribc2 and Rib <= Ribc3:
                    ZL = Rib * sum(bet2 * coefficients2[2, :])
                elif Rib > Ribc3 and Rib <= Ribc4:
                    ZL = Rib * sum(bet2 * coefficients2[3, :])
                elif Rib > Ribc4 and Rib <= Ribc5:
                    ZL = Rib * sum(bet2 * coefficients2[4, :])
                elif Rib > Ribc5 and Rib <= Ribc6:
                    ZL = Rib * sum(bet2 * coefficients2[5, :])
                elif Rib > Ribc6:
                    ZL = Rib * sum(bet2 * coefficients2[6, :])

            elif zz / z0m <= 10 ** 5 and zz / z0m > 40 and z0m / z0h > 10 ** 11 and z0m / z0h <= 1.07 * 10 ** 13:
                # print(region 8")
                coefficients1 = np.array([[0, 0, 0, 0.05139, -0.02991, 0.004664, -2.135 * 10 ** -4, 6.535 * 10 ** -5],
                                          [0, 0, 0, 0.04919, -0.0197, 0.002011, -3.325 * 10 ** -4, 7.974 * 10 ** -5],
                                          [0.5775, -0.2236, 0.03477, 0.03805, -0.01617, 0.00177, -2.191 * 10 ** -5,
                                           1.067 * 10 ** -5]])
                coefficients2 = np.array([[-3.13, 5.26, -29.85, 57.04, 0.2176, -0.00898, -1.756, -1.663, -0.007271,
                                           0.0304, 0.05349, 8.978 * 10 ** -5, -6.252 * 10 ** -4, 0.9846, -1.011, 14.45,
                                           4.433, -0.05083, -0.2604, -0.2977, 0.001361, 0.00375, -1.464 * 10 ** -5,
                                           -0.004659, 0.6393, 0, 0, 0, 0, 8.014 * 10 ** -4, -0.01934, 0],
                                          [-49.55, 97.14, 352.5, -573.4, 2.052, -21.41, 13.12, 20.82, 0.1357, 0.238,
                                           -0.7316, -0.003367, 0.006023, 14.57, 0, 0, -54.39, -0.8911, 1.478, 2.13,
                                           -9.36 * 10 ** -4, -0.04272, 1.939 * 10 ** -4, -1.165, 0, -3.616, 0.06747,
                                           0.01581, -3.126 * 10 ** -4, 0.03485, 0, -0.001713],
                                          [0, 0, 10.72, 0, 0, 0, 0, -1.354, -0.06227, 0, 0.08799, 0.002359, -0.002387,
                                           -0.2492, 19.79, -18.86, 9.463, 0, 0, -0.3291, 0, 0.01369, -2.14 * 10 ** -4,
                                           0, -1.689, 1.036, 0.00194, -0.02897, 8.316 * 10 ** -4, 0.01694, 0.06734,
                                           -0.001447],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, -0.01477, -0.001292, 0, 3.921 * 10 ** -4, 0, 0,
                                           -0.8522, 0.1065, 0, 0.374, 0.004036, 0.002528, -0.006853, -8.747 * 10 ** -5,
                                           0, -0.4307, 0.01469, 0.001642, 0, 0, 0, 0.01348, 0]])

                Ribc1 = sum(bet1 * coefficients1[0, :])
                Ribc2 = sum(bet1 * coefficients1[1, :])
                Ribc3 = sum(bet1 * coefficients1[2, :])

                if Rib > 0 and Rib <= Ribc1:
                    ZL = Rib * sum(bet2 * coefficients2[0, :])
                elif Rib > Ribc1 and Rib <= Ribc2:
                    ZL = Rib * sum(bet2 * coefficients2[1, :])
                elif Rib > Ribc2 and Rib <= Ribc3:
                    ZL = Rib * sum(bet2 * coefficients2[2, :])
                elif Rib > Ribc3:
                    ZL = Rib * sum(bet2 * coefficients2[3, :])
        # for unstable conditions
        elif Rib < 0:
            coefficients3 = np.array([[-116.93, 0, 0, 0, 126.9, 0, 0, 0, 0, -115.09, 111.64, -3.572, 3.8304, -4.5297,
                                       25.776, 0.45351, -26.473, 6.6361, -3.3234, 0, 0, -0.65868, 1.0393],
                                      # 这个系数跟书上的怎么不一样？
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14.904, 0, 0, 6.8022, 2.6097, -6.1598, -2.5799,
                                       -0.98398, 0, 0, -0.19337, 0.87712],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 60.525, -128.39, 0, 0, 0, 0, 33.947, 0, 5.4086, -4.91,
                                       1.516, -6.877, -0.21653, 1.2529],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4.5577, 5.4274, 0, 0, -1.5934,
                                       -0.46985, -1.4933, -0.184, 1.0526],
                                      [0, 0, 0, 0, 133.3, 0, 12.548, 0, -34.644, -121.31, 0, -4.1189, 0, 0, 26.6,
                                       33.905, 0, 3.2532, -3.4678, 0, -8.6705, 0, 1.3343],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, -21.221, 9.7024, 0, 0, 5.4758, 23.793, 2.8282,
                                       -12.674, 0, -1.3558, -1.7794, 0, -0.42907, 1.0925],
                                      [0, 0, 0, 0, 0, 0, 0, 0, -343.68, -637.21, 935.47, 0, 0, 0, 152.13, 85.433,
                                       -212.9, 1.6941, 0, 0, -5.0047, -0.50742, 1.4164],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, -35.397, 0, 0, 0, 0, 32.746, 10.744, -19.461, 0, 0, 0,
                                       -3.3425, -0.53463, 1.3388]])
            if Rib < 0 and Rib > -2 and zz / z0m <= 80 and zz / z0m >= 10 and z0m / z0h >= 0.607 and z0m / z0h <= 10:
                # print(region 9")
                ZL = Rib * Lom ** 2 / Loh * sum(bet3 * coefficients3[0, :])
            elif Rib < 0 and Rib > -2 and zz / z0m <= 80 and zz / z0m >= 10 and z0m / z0h > 10 and z0m / z0h <= 1.069 * 10 ** 13:
                # print(region 10")
                ZL = Rib * Lom ** 2 / Loh * sum(bet3 * coefficients3[1, :])
            elif Rib < 0 and Rib > -2 and zz / z0m <= 10 ** 5 and zz / z0m > 80 and z0m / z0h >= 0.607 and z0m / z0h <= 10:
                # print(region 11")
                ZL = Rib * Lom ** 2 / Loh * sum(bet3 * coefficients3[2, :])
            elif Rib < 0 and Rib > -2 and zz / z0m <= 10 ** 5 and zz / z0m > 80 and z0m / z0h > 10 and z0m / z0h <= 1.069 * 10 ** 13:
                # print(region 12")
                ZL = Rib * Lom ** 2 / Loh * sum(bet3 * coefficients3[3, :])
            elif Rib <= -2 and Rib > -5 and zz / z0m <= 80 and zz / z0m >= 10 and z0m / z0h >= 0.607 and z0m / z0h <= 10:
                # print(region 13")
                ZL = Rib * Lom ** 2 / Loh * sum(bet3 * coefficients3[4, :])
            elif Rib <= -2 and Rib > -5 and zz / z0m <= 80 and zz / z0m >= 10 and z0m / z0h > 10 and z0m / z0h <= 1.069 * 10 ** 13:
                # print(region 14")
                ZL = Rib * Lom ** 2 / Loh * sum(bet3 * coefficients3[5, :])
            elif Rib <= -2 and Rib > -5 and zz / z0m <= 10 ** 5 and zz / z0m > 80 and z0m / z0h >= 0.607 and z0m / z0h <= 10:
                # print(region 15")
                ZL = Rib * Lom ** 2 / Loh * sum(bet3 * coefficients3[6, :])
            elif Rib <= -2 and Rib > -5 and zz / z0m <= 10 ** 5 and zz / z0m > 80 and z0m / z0h > 10 and z0m / z0h <= 1.069 * 10 ** 13:
                # print(region 16")
                ZL = Rib * Lom ** 2 / Loh * sum(bet3 * coefficients3[7, :])

        if ZL < 0:
            zxx = (1.0 - cgammam * ZL) ** calpham
            zyy = (1.0 - cgammah * ZL) ** calphah
            zpsim = 2.0 * np.log((1.0 + zxx) / 2.0) + np.log((1.0 + zxx ** 2) / 2.0) - 2.0 * math.atan(zxx) + p2
            zpsih = 2.0 * np.log((1.0 + zyy) / 2.0)
            zxx0 = (1.0 - cgammam * ZL / zzz0m) ** calpham
            zyy0 = (1.0 - cgammah * ZL / zzz0h) ** calphah
            zpsim0 = 2.0 * np.log((1.0 + zxx0) / 2.0) + np.log((1.0 + zxx0 ** 2) / 2.0) - 2.0 * math.atan(zxx0) + p2
            zpsih0 = 2.0 * np.log((1.0 + zyy0) / 2.0)
            ZLmcorr = ZL * zfacM2
            ZLhcorr = ZL * zfacH2
            zphistarm = (1.0 - cgammam * ZLmcorr) ** (-calpham)
            zphistarh = (1.0 - cgammah * ZLhcorr) ** (-calphah)
        elif ZL > 0:
            zpsim = -caa * np.log(ZL + (1.0 + ZL ** cbb) ** (1.0 / cbb))
            zpsih = -ccc * np.log(ZL + (1.0 + ZL ** cdd) ** (1.0 / cdd))
            zpsim0 = -caa * np.log(ZL / zzz0m + (1.0 + (ZL / zzz0m) ** cbb) ** (1.0 / cbb))
            zpsih0 = -ccc * np.log(ZL / zzz0h + (1.0 + (ZL / zzz0h) ** cdd) ** (1.0 / cdd))
            ZLmcorr = ZL * zfacM2
            ZLhcorr = ZL * zfacH2
            zphistarm = 1.0 + caa * (ZLmcorr + ZLmcorr ** cbb * (1.0 + ZLmcorr ** cbb) ** ((1.0 - cbb) / cbb)) / (
                        ZLmcorr + (1.0 + ZLmcorr ** cbb) ** (1.0 / cbb))
            zphistarh = 1.0 + ccc * (ZLhcorr + ZLhcorr ** cdd * (1.0 + ZLhcorr ** cdd) ** ((1.0 - cdd) / cdd)) / (
                        ZLhcorr + (1.0 + ZLhcorr ** cdd) ** (1.0 / cdd))
        elif np.isnan(ZL) or ZL == 0:
            zphistarm = np.nan
            zphistarh = np.nan
            zpsim = np.nan
            zpsim0 = np.nan
            zpsih = np.nan
            zpsih0 = np.nan
        zpsistarm = zphistarm * zfacM
        zpsistarh = zphistarh * zfacH
        CM = 0.16 / ((zL0M - zpsim + zpsim0 + zpsistarm) ** 2)
        CH = 0.16 / ((zL0H - zpsih + zpsih0 + zpsistarh) * (zL0M - zpsim + zpsim0 + zpsistarm))

    # --------------------------------------求波导高度-----------------------------------------#
    tao = CM * rhoa * u * u
    # print("u", u)
    # print("CH ", CH)
    # print("cpa ", cpa)
    # print("thetas ", thetas)
    # print("theta ", theta)
    Hs = CH * rhoa * u * cpa * (thetas - theta)
    # print("Hs ", Hs)
    # print("Le ", Le)
    # print("dq ", dq)
    Hl = CH * rhoa * u * Le * dq
    E = B / (rhoa * cpa)
    F = C / (rhoa * Le)
    if ZL > 0:
        kk = 0
        HH = 1.0
        HH1 = 1.0 * 1000 + 50;
        while abs(HH - HH1) > abs(0.001 * HH) and kk < 20:
            kk = kk + 1
            HH1 = HH
            HH = (E * Hs + F * Hl) / (von * (A + 0.157) * np.abs(np.sqrt(tao / rhoa))) * \
                 (1 + 5.3 * (HH / zz * ZL + (HH / zz * ZL) ** 1.1 * (1 + (HH / zz * ZL) ** 1.1) ** -0.091) /
                  (HH / zz * ZL + (1 + (HH / zz * ZL) ** 1.1) ** 0.91))
    else:
        kk = 0
        HH = 1.0
        HH1 = 1.0 * 1000 + 50
        while np.abs(HH - HH1) > np.abs(0.001 * HH) and kk < 20:
            kk = kk + 1
            HH1 = HH
            HH = (E * Hs + F * Hl) / (
                        von * (A + 0.157) * np.abs(np.sqrt(tao / rhoa)) * abs(np.sqrt(1 - 16 * HH / zz * ZL)))
    if HH.real > 40:
        HH = 40
    elif HH.real < 0 or np.isnan(HH):
        HH = 0
    return HH

def Interplot(data, start=50, end=3000, nums=296, kind="cubic"):
    data = np.array(data)
    newdata = []
    xnew = np.linspace(start, end, nums, endpoint=True)
    newdata.append(xnew)
    for i in range(data.shape[1]):
        if i == 0:
            continue
        line = interp1d(data[:, 0], data[:, i], kind=kind)
        ynew = line(xnew)
        newdata.append(ynew)
    data = np.array(newdata).transpose()
    return data

def generate_data_diancichuanbo(ref, h):
    pre_data=np.zeros((5, ref.shape[1]))
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
        for j in range(ref.shape[0]-1):
            if r[j, i] <= 0 and r[j+1, i] > 0 and r[0, i] < 0:
                hh1[n, i] = h[j, i]
                g1[n, i] = h[j, i]
                ddm1[n, i] = ref[0, i]-ref[j, i]
                tidu1[n, i] = ddm1[n, i]/hh1[n, i]
                if g1[n, i] < 40:
                    flag1[n, i] = 1
                else:
                    flag1[n, i] = 2
                n += 1
            elif r[j+1, i] <= 0 and r[j, i] > 0 and r[0, i] > 0:
                delm2[k, i] = ref[j, i]
                for ss in range(ref.shape[0]-j-1):
                    if r[j+ss, i] <=0 and  r[j+ss+i, i] > 0:
                        if ref[j+ss, i] > ref[0, i]:
                            flag2[k, i] = 3
                            g2[k, i] = h[j, i]
                            ddm2[k, i] = ref[j+ss, i]-delm2[k ,i]
                            for b in range(j-1):
                                if np.abs(h[b, i]- h[j, i]) < 0.1:
                                    hh2[k, i] = h[j+ss, i] - g2[k, i]
                                    tidu2[k, i] = ddm2[k, i]/hh2[k, i]
                    elif ref[j+ss, i] < ref [0, i]:
                        flag2[k, i] = 4
                        ddm2[k, i] = ref[j+ss, i] - delm2[k, i]
                        hh2[k, i] = h[j+ss, i] - h[j, i]
                        tidu2[k, i] = delm2[k, i]/g2[k, i]
                cc1[k, i] =delm2[k, i]/g2[k, i]
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
                        q=zt
                        break
                pre_data[0][i]=m
                pre_data[1][i] = zt
                pre_data[2][i] = c1
                pre_data[3][i]=zb
                pre_data[4][i]=q
                break
            elif(flag1[j][i] == 2):
                #发生表面波导1
                zb=0
                q=0
                c1=0
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
            if(flag2[j][i] == 3):
                #发生了悬空波导
                m=0
                zt=0
                c1=0
                zb=0
                q=0
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
            elif(flag2[j][i] == 4):
                #发生表面波导2
                q=0
                c1=0
                zb=0
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

def dianciLoss(ref, h):
    pre_data = generate_data_diancichuanbo(ref, h)
    loss = SP.spe(pre_data[0][0], pre_data[1][0], pre_data[2][0], pre_data[3][0], pre_data[4][0])
    loss = np.array(loss)
    return loss

def LeiDa(Text1, Text2, Text3, Text4, Text5, Text6, Text7, Text8, Text9, Text10, Text11, Text12):
    Lossflag=135.43+10*math.log10(float(Text2)*float(Text12)*float(Text1)*float(Text1))-10*math.log10(float(Text8))+2*float(Text4)-float(Text9)-float(Text10)-10*math.log10(float(Text7))
    Lossflag = Lossflag * 0.5
    return Lossflag

# 陷获频率与截至波长
#悬空波导陷获频率和截止波长计算
#input M_top M_bottom,d单位km
#output f陷获频率 单位GHz m截止波长单位m
def xkTrap(M_top,M_bottom,d):
    f = 79.4945/(math.sqrt(M_bottom-M_top)*d*1000)
    m = 2.9979*math.pow(10, -1)/f
    return float(f), float(m)

#表面波导陷获频率和截止波长计算
#input M_0 M_top,d单位km
#output f陷获频率 单位GHz m截止波长单位m
def bmTrap(M_0,M_top,d):
    f = 119.2417 / (math.sqrt(M_0 - M_top) * d * 1000)
    m = 2.9979*math.pow(10, -1)/f
    return float(f), float(m)

def getTrap(xResult, bResult):
    if xResult[0][0] != 0:
        return xkTrap(xResult[2][0], xResult[3][0], xResult[7][0])
    elif bResult[0][0] != 0:
        return bmTrap(bResult[1][0], bResult[2][0], bResult[3][0])
    else:
        return None, None