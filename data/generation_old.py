#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import random
import time
# generate temperature, humidity, pressure, wind, direction, time, position
database = []
class Generation():
    database = []
    def __init__(self):
        for i in range(171):
            tem, wind, press, direction, hum, times = self.generate_base()
            evap = self.generate_evap()
            temp = {
                'tem': tem,
                'wind': wind,
                'press': press,
                'direction': direction,
                'hum': hum,
                'times': times,
                'evap': evap,
            }
            self.database.append(temp)
    def generate_base(self):
        tem = random.random() * 150 + 150
        wind = random.random() * 30
        press = random.random()* 1100
        direction = random.random() * 360
        hum = random.random()
        times = time.strftime("%Y-%m-%d, %H:%M")
        return tem, wind, press, direction, hum, times

    def generate_evap(self):
        evap = random.random() * 50
        return evap

    def update(self):
        tem, wind, press, direction, hum, times = self.generate_base()
        evap = self.generate_evap()
        temp = {
            'tem': tem,
            'wind': wind,
            'press': press,
            'direction': direction,
            'hum': hum,
            'times': times,
            'evap': evap,
        }
        self.database.append(temp)