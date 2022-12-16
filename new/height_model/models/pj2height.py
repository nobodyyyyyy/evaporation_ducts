import math

def pj2height(t, rh, to, u, p, z, stable_check=False):
    '''
    PS. 原版matlab代码里面貌似返回值还有一个M变量, 但不知什么意思，因为我们这里只需要高度, M变量的计算就不考虑了
    t: 空气温度
    to: 海表温度 ts
    z: 测量高度
    u: 风速
    rh: 相对湿度
    '''

    b = -0.125 # 折射率梯度
    zo = 0.00015 # 海面粗糙长度
    beta = 5.2
    Tz = t + 273.16
    To = to + 273.16
    _stable = False

    if Tz - To <= -1:
        Rib = 98*z*(Tz-To)/(u**2*Tz) # 理查森数

        # 修正
        if Rib <= 1:
            Rib = Rib
        else:
            Rib = 1

        if Rib <= -3.75:
            Y = 0.05 # 经验剖面细数
        elif Rib >- 3.75 and Rib <= -0.12:
            Y = 0.065 + 0.004 * Rib
        elif Rib >= -0.12 and Rib <= 0.14:
            Y = 0.109 + 0.367 * Rib
        else:
            Y = 0.155 + 0.021 * Rib

        L = 10 * z * Y / Rib # M-O长度

        # 稳定或中性情况下
        es = 6.105 * math.exp(25.22 * t / Tz - 5.31 * math.log(Tz / 273.2)) # 测量高度空气饱和水汽压
        e = es * rh / 100
        eo = 6.105 * math.exp(25.22 * to / To - 5.31 * math.log(To / 273.2)) # 海面饱和水汽压
        Np = 77.6 / Tz * (1000 + 4810 / Tz * e)
        Npo = 77.6 / To * (1000 + 4810 / To * eo)
        DNp = Np - Npo

        if Rib >= 0:
            B = math.log(z / zo) + beta / L * (z - zo)
            H = DNp / (b * B - DNp * beta / L) # 波导高度
        
        # 不稳定情况下
        else:
            c = z / L
            if c >= -0.01:
                faith_c = -4.5 * c
            elif c >= -0.026 and c < -0.001:
                faith_c = 10**(1.02 * math.log(-c) + 0.69)
            elif c >= -0.1 and c < -0.026:
                faith_c = 10**(0.776 * math.log(-c) + 0.306)
            elif c >= -1 and c < -0.1:
                faith_c = 10**(0.630 * math.log(-c) + 0.16)
            elif c >= -2.2 and c < -1:
                faith_c = 10**(0.414 * math.log(-c) + 0.16)
            else:
                faith_c = 2

            B = math.log(z / zo) - faith_c
            H = ((b * B / DNp)**4 - 4 * 4.5 / L * ((b * B / DNp)**3))**(-1/4)

    # 温差修正-气海温差大于-1
    else:
        # 计算温差为-1
        to1 = to
        t1 = to1 - 1
        To1 = To
        Tz1 = To1 - 1
        Rib1 = 98 * z * (Tz1 - To1) / (u**2 * Tz1) # 理查森数
        # 参数修正
        if Rib1 <= 1:
            Rib1 = Rib1
        else:
            Rib1 = 1
        
        if Rib1 <= -3.75:
            Y1 = 0.05 # 经验剖面细数
        elif Rib1 >- 3.75 and Rib1 <= -0.12:
            Y1 = 0.065 + 0.004 * Rib1
        elif Rib1 >= -0.12 and Rib1 <= 0.14:
            Y1 = 0.109 + 0.367 * Rib1
        else:
            Y1 = 0.155 + 0.021 * Rib1

        L1 = 10 * z * Y1 / Rib1 # M-O长度

        # 稳定或中性情况下
        es1 = 6.105 * math.exp(25.22 * t1 / Tz1 - 5.31 * math.log(Tz1 / 273.2)) # 测量高度空气饱和水汽压
        e1 = es1 * rh / 100
        eo1 = 6.105 * math.exp(25.22 * to1 / To1 - 5.31 * math.log(To1 / 273.2)) # 海面饱和水汽压
        Np1 = 77.6 / Tz1 * (1000 + 4810 / Tz1 * e1)
        Npo1 = 77.6 / To1 * (1000 + 4810 / To1 * eo1)
        DNp1 = Np1 - Npo1

        if Rib1 >= 0:
            _stable = True
            B1 = math.log(z / zo) + beta / L1 * (z - zo);
            H1 = DNp1 / (b * B1 - DNp1 * beta / L1) # 波导高度
            
        # 不稳定情况下
        else:
            _stable = False
            c1 = z / L1

            if c1 >= 0.01:
                faith_cc = -4.5 * c1
            elif c1 >= -0.026 and c1 < -0.001:
                faith_cc = 10**(1.02 * math.log(-c1) + 0.69)
            elif c1 >= -0.1 and c1 < -0.026:
                faith_cc = 10**(0.776 * math.log(-c1) + 0.306)
            elif c1 >= -1 and c1 < -0.1:
                faith_cc = 10**(0.630 * math.log(-c1) + 0.16)
            elif c1 >= -2.2 and c1 < -1:
                faith_cc = 10**(0.414 * math.log(-c1) + 0.16)
            else:
                faith_cc = 2

            B1 = math.log(z / zo) - faith_cc;
            H1 = ((b * B1 / DNp1)**4 - 4 * 4.5 / L1 * ((b * B1 / DNp1)**3))**(-1/4)

        # 计算气海温差为零
        to2 = to
        t2 = to2
        To2 = To
        Tz2 = To2
        Rib2 = 98 * z * (Tz2 - To2) / (u**2 * Tz2) # 理查森数

        # 参数修正
        if Rib2 <= 1:
            Rib2 = Rib2
        else:
            Rib2 = 1

        if Rib2 <= -3.75:
            Y2 = 0.05 # 经验剖面细数
        elif Rib2 >- 3.75 and Rib2 <= -0.12:
            Y = 0.065 + 0.004 * Rib
        elif Rib2 >= -0.12 and Rib2 <= 0.14:
            Y2 = 0.109 + 0.367 * Rib2
        else:
            Y2 = 0.155 + 0.021 * Rib2

        L2 = 10 * z * Y2 / Rib2 # M-O长度

        # 稳定或中性情况下
        es2 = 6.105 * math.exp(25.22 * t2 / Tz2 - 5.31 * math.log(Tz2 / 273.2)) # 测量高度空气饱和水汽压
        e2 = es2 * rh / 100
        eo2 = 6.105 * math.exp(25.22 * to2 / To2 - 5.31 * math.log(To2 / 273.2)) # 海面饱和水汽压
        Np2 = 77.6 / Tz2 * (1000 + 4810 / Tz2 * e2)
        Npo2 = 77.6 / To2 * (1000 + 4810 / To2 * eo2)
        DNp2 = Np2 - Npo2

        if Rib2 >= 0:
            B2 = math.log(z / zo) + beta / L2 * (z - zo)
            H2 = DNp2 / (b * B2 - DNp2 * beta / L2) # 波导高度

        # 不稳定情况下
        else:
            c2 = z / L2

            if c2 >= 0.01:
                faith_ccc = -4.5 * c2
            elif c2 >= -0.026 and c2 < -0.001:
                faith_ccc = 10**(1.02 * math.log(-c2) + 0.69)
            elif c2 >= -0.1 and c2 < -0.026:
                faith_ccc = 10**(0.776 * math.log(-c2) + 0.306)
            elif c2 >= -1 and c2 < -0.1:
                faith_ccc = 10**(0.630 * math.log(-c2) + 0.16)
            elif c2 >= -2.2 and c2 < -1:
                faith_ccc = 10**(0.414 * math.log(-c2) + 0.16)
            else:
                faith_ccc = 2

            B2 = math.log(z / zo) - faith_ccc;
            H2 = ((b * B2 / DNp2)**4 - 4 * 4.5 / L2 * ((b * B2 / DNp2)**3))**(-1/4)
            

        if H1 <= H2:
            H = H1
        else:
            Rib = 98 * z * (Tz - To) / (u**2 * Tz) # 理查森数

            # 修正
            if Rib <= 1:
                Rib = Rib
            else:
                Rib = 1

            if Rib <= -3.75:
                Y = 0.05 # 经验剖面细数
            elif Rib >- 3.75 and Rib <= -0.12:
                Y = 0.065 + 0.004 * Rib
            elif Rib >= -0.12 and Rib <= 0.14:
                Y = 0.109 + 0.367 * Rib
            else:
                Y = 0.155 + 0.021 * Rib  

            L = 10 * z * Y / Rib # M-O长度  

            # 稳定或中性情况下
            es = 6.105 * math.exp(25.22 * t / Tz - 5.31 * math.log(Tz / 273.2)) # 测量高度空气饱和水汽压
            e = es * rh / 100
            eo = 6.105 * math.exp(25.22 * to / To - 5.31 * math.log(To / 273.2)) # 海面饱和水汽压
            Np = 77.6 / Tz * (1000 + 4810 / Tz * e)
            Npo = 77.6 / To * (1000 + 4810 / To * eo)
            DNp = Np - Npo

            if Rib >= 0:
                B = math.log(z / zo) + beta / L * (z - zo)
                H = DNp / (b * B - DNp * beta / L) # 波导高度

            # 不稳定情况下
            else:
                c = z / L

                if c >= -0.01:
                    faith_c = -4.5 * c
                elif c >= -0.026 and c < -0.001:
                    faith_c = 10**(1.02 * math.log(-c) + 0.69)
                elif c >= -0.1 and c < -0.026:
                    faith_c = 10**(0.776 * math.log(-c) + 0.306)
                elif c >= -1 and c < -0.1:
                    faith_c = 10**(0.630 * math.log(-c) + 0.16)
                elif c >= -2.2 and c < -1:
                    faith_c = 10**(0.414 * math.log(-c) + 0.16)
                else:
                    faith_c = 2    

                B = math.log(z / zo) - faith_c
                H = ((b * B / DNp)**4 - 4 * 4.5 / L * ((b * B / DNp)**3))**(-1/4)

    if H > 40:
        H = 40

    elif H < 0:
        H = 0

    if stable_check:
        return H, _stable
    else:
        return H

