#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import scipy.io as scio
import numpy as np
import random

a = np.random.rand(3, 5)

print(a)
print(a[:, [0, 2]])

print([False for i in range(5)])
for i in range(10):
    print(random.randint(0, 2))