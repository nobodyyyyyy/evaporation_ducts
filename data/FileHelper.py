#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from data.generation import is_number

class dataReader():
    def __init__(self):
        self.dataset = []
        self.filename = None

    def readTxt(self, metadata, filename, limit=3):
        self.dataset = []
        l = 0
        for line in str(metadata, 'ANSI').split("\n"):
            line = line.strip('\n')
            if l >= 2:
                temp = line.split()
                temperature = float(temp[2])
                press = float(temp[3])
                humidity = float(temp[4])
                altitude = float(temp[7]) / 1000
                direction = float(temp[5])
                wind = float(temp[6])
                if altitude > limit:
                    break
                self.dataset.append([altitude, temperature, press, humidity, wind, direction])
            l = l + 1
        self.filename = filename

    def readTpu(self, metadata, filename, limit=3):
        self.dataset = []
        l = 0
        for line in str(metadata, 'utf-8').split('\n'):
            line = line.strip('\n')
            if l >= 4:
                temp = line.split()
                if is_number(temp[1]) and is_number(temp[2]) and is_number(temp[3]) and is_number(temp[5]):
                    temperature = float(temp[1])
                    press = float(temp[3])
                    humidity = float(temp[2])
                    altitude = float(temp[5]) / 1000
                    if altitude > limit:
                        break
                    self.dataset.append([altitude, temperature, press, humidity])
            l = l + 1
        self.filename = filename
