#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from statsmodels.tsa.arima_model import ARIMA
import torch.nn as nn
from utils import Stationary_A
import statsmodels.api as sm

class ARIMA_model(nn.Module):
    def __init__(self, q=None, p=None, d=None):
        super(ARIMA_model, self).__init__()
        self.q = q
        self.p = p
        self.d = d
        self.stationary = False
        self.properModel = None

    def train(self, x):
        if self.q is not None and self.d is not None and self.p is not None:
            model = ARIMA(x, [self.p, self.d, self.q])
        else:
            if self.d is None:
                flag, dif = Stationary_A(x)
                self.stationary = flag
                self.d = dif
            best_pq_list = []
            # bic优先
            (p_aic, q_aic) = sm.tsa.arma_order_select_ic(x, max_ar=min(int(len(x) * 0.2), 6), max_ma=min(int(len(x) * 0.2), 4), ic='aic')['aic_min_order']
            best_pq_list.append([p_aic, q_aic])
            (p_bic, q_bic) = sm.tsa.arma_order_select_ic(x, max_ar=min(int(len(x) * 0.2), 6), max_ma=min(int(len(x) * 0.2), 4), ic='bic')['bic_min_order']
            best_pq_list.append([p_bic, q_bic])
            (p_hqic, q_hqic) = sm.tsa.arma_order_select_ic(x, max_ar=min(int(len(x) * 0.2), 6), max_ma=min(int(len(x) * 0.2), 4), ic='hqic')['hqic_min_order']
            best_pq_list.append([p_hqic, q_hqic])
            best_pq_set = {}
            for i in range(len(best_pq_list)):
                j = i + 1
                best_pq_set[best_pq_list[i]] = 1
                while j < len(best_pq_list):
                    if best_pq_list[i] == best_pq_list[j]:
                        best_pq_set[best_pq_list[i]] += 1
                    j += 1
            max = 0
            for key in best_pq_set:
                if max < best_pq_set[key]:
                    max = best_pq_set[key]
                    best_pq = key
            if max == 1:
                best_pq = best_pq_list[1]
            self.p = best_pq[0]
            self.q = best_pq[1]
            model = ARIMA(x, [self.p, self.d, self.q])
        self.properModel = model.fit()

    def forward(self):
        result = self.properModel.predict(1, dynamic=True)
        return result






