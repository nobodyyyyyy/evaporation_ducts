from cmath import exp, log, atan, sqrt

import numpy as np


def R_S(t: np.array, p: np.array):
    return 6.112 * exp(17.502 * t / (t + 241)) * (1.0007 + 3.46e-6 * p)


def qsee(ts, P):
    es = R_S(ts, P) * 0.98
    qs = es * 0.622 / (P - 0.378 * es)
    return qs


def psiu_nps(zet):
    # fixme 我就把 zet 看成一个值了，有问题再说

    if zet.real <= 0:
        c = min(50., (.35 * zet).real)
        psi = -((1+2/3*zet) * 1.5+2/3*(zet-14.28) / exp(c)+8.525)
        return psi

    x = (1 - 15 * zet) * .25
    psik = 2 * log((1 + x) / 2) + log((1 + x * x) / 2) - 2 * atan(x) + 2 * atan(1)
    x = (1 - 10 * zet) * .3333
    psic = 1.5 * log((1 + x + x * x) / 3) - sqrt(3) * atan((1 + 2 * x) / sqrt(3)) + 4 * atan(1) / sqrt(3)
    f = zet * zet / (1 + zet * zet)
    psi = (1 - f) * psik + f * psic
    return psi


def psit_nps(zet):
    # fixme 同上

    if zet.real <= 0:
        c = min(50., (.35 * zet).real)
        psi = -((1+2/3*zet) * 1.5+2/3*(zet-14.28) / exp(c)+8.525)
        return psi

    x = (1 - 15 * zet) * .5
    psik = 2 * log((1 + x) / 2)
    x = (1 - 34 * zet) * .3333
    psic = 1.5 * log((1 + x + x * x) / 3) - sqrt(3) * atan((1 + 2 * x) / sqrt(3)) + 4 * atan(1) / sqrt(3)
    f = zet * zet / (1 + zet * zet)
    psi = (1 - f) * psik + f * psic
    return psi


def psiu_25(zet):

    if zet.real <= 0:
        psi = -5 * zet
        return psi

    x = (1 - 16 * zet) * .25
    psik = 2 * log((1 + x) / 2)+ log((1 + x * x) / 2) - 2 * atan(x) + 2 * atan(1)
    x= (1 - 12.87 * zet) * .3333
    psic = 1.5 * log((1 + x + x * x) / 3) - sqrt(3) * atan((1 + 2 * x) / sqrt(3)) + 4 * atan(1) / sqrt(3)
    f = zet * zet / (1 + zet * zet)
    psi = (1 - f) * psik + f * psic

    return psi


def psit_25(zet):
    return psit_nps(zet)


def fait(zet):
    x = (1 - 16 * zet)**0.5
    y = (1 - 12.87 * zet)**0.3333
    f = zet * zet / (1 + zet * zet)
    psi = (1 - f) / x + f / y
    return psi
