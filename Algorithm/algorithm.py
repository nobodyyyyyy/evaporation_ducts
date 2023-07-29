#!/usr/bin/env python
# -*- coding:utf-8 -*-
import time
import threading
import pandas
import numpy as np
from cmath import exp
import matlab.engine
from scipy.interpolate import interp1d
# import SP
import math


# 可以优化的部分
# -----------------------------------
# 计算下面部分的耗时，如果耗时比较久，则可以优化
# engine = matlab.engine.start_matlab()

# 结果：
# 通过计算结果可得知，上部分代码耗费的时间为5.5s左右，有极大的优化价值
# -----------------------------------


# 通过海表温计算的蒸发波导
# 输入为5个参数海温、气温、湿度、风速、压强
# 此data输入的内容为 [海温，气压，1层气温，1层湿度，1层风向，1层风速，
# 2层气温，2层湿度，2层风向，2层风速，3层气温，3层湿度，3层风向，3层风速]
# layer为层数，根据excel的说明，共有1 2 3 三层
# data为numpy型数据！！！否则下面运行不了
# def zhengfaMs(data, layer=0):
#     start = time.time()
#     engine = matlab.engine.start_matlab()
#     end = time.time()
#     print("time cost:", end - start)
#     evap = []
#     for i in range(data.shape[0]):
#         # 气温、海表温、湿度、风速、压强
#         temp = engine.demo(matlab.double([data[i][2 + 4 * layer]]), matlab.double([data[i][0]]),
#                            matlab.double([data[i][3 + 4 * layer]]),
#                            matlab.double([data[i][5 + layer * 4]]), matlab.double([data[i][1]]))
#         # print(temp)
#         evap.append(temp)
#     return evap


# p气压 t温度，shidu相对湿度，z高度,计算得到某一高度的大气折射率
def zheshelv(t, p, shidu, z):
    # atmospheric_refractive_index_M
    # e为饱和水汽压
    e = 6.112 * exp(17.67 * t / (t + 243.5))
    e = e * shidu
    M = 77.6 * p / t + 3.73 * pow(10, 5) * e / (t * t) + 0.157 * z
    return M.real


# 输入data，生成其他算法需要的廓线和高度数据
def generate_data(data):
    # data=np.load('gaokong.npy')
    h = np.zeros((data.shape[0], 1))
    ref = np.zeros((data.shape[0], 1))
    for i in range(0, data.shape[0]):
        h[i][0] = data[i][0]
        ref[i][0] = zheshelv(data[i][1], data[i][2], data[i][4], data[i][0])
    return ref, h


# 主要服务于，使用.tpu的snd数据
# 数据格式为 [alt(高度), tem(温度), pre(压强), hum(湿度)]
def generate_datas(data):
    h = np.zeros((data.shape[0], 1))
    ref = np.zeros((data.shape[0], 1))
    for i in range(0, data.shape[0]):
        h[i][0] = data[i][0]
        ref[i][0] = zheshelv(data[i][2], data[i][1], data[i][3], data[i][0])
    return ref, h


# 输出表面和悬空波导信息
# selected = "n"表示用gaokong.txt格式的数据
# selected = "s"表示用snd*.tpu格式的数据
def xk_bmM(data, selected="n"):
    if selected == "n":
        ref, h = generate_data(data)
    else:
        ref, h = generate_datas(data)
    engine = matlab.engine.start_matlab()
    # ref, h = engine.loaddata(nargout=2)
    # ref = np.array(ref)
    # h = np.array(h)
    ref = matlab.double(ref.tolist())
    h = matlab.double(h.tolist())
    xuankongM, biaomianM = engine.xbM(ref, h, nargout=2)
    xuankongM = np.array(xuankongM)
    biaomianM = np.array(biaomianM)
    if (xuankongM[0][0] != 0) & (biaomianM[0][0] != 0):
        return xuankongM, biaomianM
    elif (xuankongM[0][0] != 0) & (biaomianM[0][0] == 0):
        return xuankongM, []
    elif (xuankongM[0][0] == 0) & (biaomianM[0][0] != 0):
        return [], biaomianM
    else:
        return [], []


# 输出蒸发波导高度
# selected 同上
def zhengfaM(data, selected="n"):
    if selected == "n":
        ref, h = generate_data(data)
    else:
        ref, h = generate_datas(data)
    engine = matlab.engine.start_matlab()
    # ref, h = engine.loaddata(nargout=2)
    # ref = np.array(ref)
    # h = np.array(h)
    ref = matlab.double(ref.tolist())
    h = matlab.double(h.tolist())
    # 输出波导特征参数
    tidu1, tidu2, g1, g2, hh1, hh2, ddm1, ddm2, flag1, flag2, cc1 = engine.ducttt(ref, h, nargout=11)
    tidu1 = np.array(tidu1)
    tidu2 = np.array(tidu2)
    g1 = np.array(g1)
    g2 = np.array(g2)
    hh1 = np.array(hh1)
    hh2 = np.array(hh2)
    ddm1 = np.array(ddm1)
    ddm2 = np.array(ddm2)
    flag1 = np.array(flag1)
    flag2 = np.array(flag2)
    cc1 = np.array(cc1)
    ref = np.array(ref)
    res = np.zeros((2, ref.shape[1]))
    count = 0
    # 判断给定的廓线集中是否存在蒸发波导，并输出蒸发波导高度
    for i in range(0, flag1.shape[1]):
        for j in range(0, flag1.shape[0]):
            if (flag1[j][i] == 1):
                # 判定发生了蒸发波导，输出蒸发波导高度
                for k in range(0, g1.shape[0]):
                    if (g1[k][i] != 0):
                        res[0][count] = i + 1
                        res[1][count] = g1[k][i]
                        count = count + 1
                        break
                break
    if (count != 0):
        return res[1]
    else:
        return []


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


# 函数输出电磁传播损耗模块参数  m:负折射指数, zt:陷获层厚度, c1:混合层斜率, zb:陷获层底厚度, q:蒸发波导厚度
def generate_data_diancichuanbo(data, selected="n"):
    engine = matlab.engine.start_matlab()
    if selected == "n":
        ref, h = generate_data(data)
    else:
        ref, h = generate_datas(data)
    pre_data = np.zeros((5, ref.shape[1]))
    ref = matlab.double(ref.tolist())
    h = matlab.double(h.tolist())
    # 输出波导特征参数
    tidu1, tidu2, g1, g2, hh1, hh2, ddm1, ddm2, flag1, flag2, cc1 = engine.ducttt(ref, h, nargout=11)
    tidu1 = np.array(tidu1)
    tidu2 = np.array(tidu2)
    g1 = np.array(g1)
    g2 = np.array(g2)
    hh1 = np.array(hh1)
    hh2 = np.array(hh2)
    ddm1 = np.array(ddm1)
    ddm2 = np.array(ddm2)
    flag1 = np.array(flag1)
    flag2 = np.array(flag2)
    cc1 = np.array(cc1)
    ref = np.array(ref)
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


# 输出list，list中存储每条廓线的电磁传播损耗
# 一般情况只计算一条廓线，同一时刻只计算一次廓线情况
# def dianciLoss(data, selected="n"):
#     pre_data = generate_data_diancichuanbo(data, selected)
#     loss = SP.spe(pre_data[0][0], pre_data[1][0], pre_data[2][0], pre_data[3][0], pre_data[4][0])
#     # SP.spe(0.137, 100.0, -0.043, 300, 0, loss)
#     loss = np.array(loss)
#     return loss


# Text1:雷达频率(MHz)
# Text2:雷达峰值功率(KW)
# Text3:天线高度(m)
# Text4:天线增益(dB)
# Text5:波束宽度(deg)
# Text6:发射仰角(deg)
# Text7:最小信噪比(dB)
# Text8:接收机带宽(MHz)
# Text9:系统综合损耗(dB)
# Text10:接收机噪声系数(dB)
# Text11:目标高度(m)
# Text12:目标散射截面(m*m)
# 输出lossflag
def LeiDa(Text1, Text2, Text3, Text4, Text5, Text6, Text7, Text8, Text9, Text10, Text11, Text12):
    Lossflag = 135.43 + 10 * math.log10(float(Text2) * float(Text12) * float(Text1) * float(Text1)) - 10 * math.log10(
        float(Text8)) + 2 * float(Text4) - float(Text9) - float(Text10) - 10 * math.log10(float(Text7))
    Lossflag = Lossflag * 0.5
    return Lossflag


if __name__ == "__main__":
    # data = np.load("snd.npy")
    # print(dianciLoss(data, "s"))
    data = np.load("snd.npy")
    X, B = xk_bmM(data, "s")
    Z = zhengfaM(data, "s")
    print("悬空波导", X)
    print("表面波导", B)
    print("蒸发波导", Z)
    # data = np.load("excel.npy")
    # evap = zhengfaMs(data, 2)
    # np.save("evap2", evap)
    # data = np.load("..\\data\\gaokong.npy")
    # # print(data)
    # data = Interplot(data)
    # # print(data)

    # newdata = []
    # # xnew = np.linspace(50, 3000, 119, endpoint=True)
    # xnew = np.linspace(50, 3000, 296, endpoint=True)
    # newdata.append(xnew)
    # for i in range(data.shape[1]):
    #     if i == 0:
    #         continue
    #     line = interp1d(data[:, 0], data[:, i], kind="cubic")
    #     ynew = line(xnew)
    #     newdata.append(ynew)
    # data = np.array(newdata).transpose()
    #
