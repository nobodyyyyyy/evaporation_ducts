#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, DateTime, Float, select
from database_table import args, engine, BasicMeteo, EvapInfor, SeInfor
import threading
import xlrd
from xlrd import xldate_as_tuple
from datetime import datetime
import random
import time

class Cal_future(threading.Thread):
    def __init__(self, func, args=()):
        super(Cal_future, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(self.args)

    def get_result(self):
        threading.Thread.join(self)
        try:
            return self.result
        except Exception:
            return None

class DataUpdate:
    def __init__(self, username='root', password='SEU440CILab!', address='223.3.89.186', port='3306', database='MeteoData', lim=300):
        # mysql用户名，mysql数据库密码，数据库IP地址，数据库端口号，数据库名
        engine = create_engine(
            "mysql+pymysql://{}:{}@{}:{}/{}".format(username, password, address, port, database))
        # 建立与数据库的会话连接
        Session_class = sessionmaker(bind=engine)
        # 这里创建一个会话实例
        self.session = Session_class()
        # 读取最近的limit条数据
        self.limit = lim

    # 查询最近limit个基本气象数据
    def Query_Basic(self):
        result = session.query(BasicMeteo).order_by(BasicMeteo.Id.desc()).limit(self.limit).all()
        return result

    # 查询时间范围在start_time 到 end_time之间的数据
    def Query_Evap(self, start_time, end_time):
        result = session.query(EvapInfor).filter(EvapInfor.Time.between(start_time, end_time))
        return result

    # 基本近海面气象数据更新
    def Update_Basic(self, times, temperature, humidity, press, direction, wind, sst=None):
        if sst is None:
            newdata = BasicMeteo(Time=times, Temperature=temperature, Humidity=humidity, Press=press,
                          Direction=direction, Wind=wind)
        else:
            newdata = BasicMeteo(Time=times, Temperature=temperature, Humidity=humidity, Press=press,
                                 Direction=direction, Wind=wind, SST=sst)
        self.session.add(newdata)
        try:
            session.commit()
            print("basic commit success!")
        except BaseException:
            print("basic commit error!")

    def Update_Evap(self, times, evap_height, futu_height):
        newdata = EvapInfor(Time=times, Evap_height=evap_height, Futu_height=futu_height)
        self.session.add(newdata)
        try:
            session.commit()
            print("evap commit success!")
        except BaseException:
            print("evap commit error!")

    def Update_SE(self, times, surf_height=0, surf_strength=0, elev_bottom=0, elev_top=0, elev_strength=0):
        newdata = SeInfor(Time=time, Surf_height=surf_height, Surf_strength=surf_strength, Elev_bottom=elev_bottom, Elev_top=elev_top, elev_strength=elev_strength)
        self.session.add(newdata)
        try:
            session.commit()
            print("se commit success!")
        except BaseException:
            print("se commit error!")


class Updata(threading.Thread):
    database = []  # 内存中的数据
    limit = 300  # 上限是300条
    Location = 0 # 当前读取到的数据库的位置
    Len = 0 # database的当前值
    elec = None
    radar = None
    def __init__(self, filepath, session, model=0, lim=300):
        threading.Thread.__init__(self)
        self.limit = lim
        self.model = model
        self.session = session
        result = session.query(BasicMeteo).order_by(BasicMeteo.Id.desc()).limit(self.limit).all()
        print('result', result)
        lens = len(result)
        print("lens", lens)
        self.Len = lens
        if lens == 0:
            self.Location = 0
        else:
            self.Location = result[lens-1]['Id']
        while lens > 0:
            lens -= 1
            self.database.append(result[lens])
        try:
            self.ws = xlrd.open_workbook(filepath)
        except:
            print('cannot find this file！')
        self.wp = self.ws.sheet_by_index(0)

    def Cal_Evap(self, data):
        # 波导可以被计算
        return random.random() * 50

    def Cal_fore(self, evap):  # 预测的方法
        return evap + self.model

    def run(self):
        nrows = self.wp.nrows
        ncols = self.wp.ncols
        startday = 1
        for i in range(nrows):
            data = []
            if i < 2:
                continue
            for j in range(ncols):
                cell = self.wp.cell_value(i, j)
                if j == 0:
                    cell = list(xldate_as_tuple(cell, 0))
                    cell[0] = 2021
                    cell[1] = 3
                    if cell[3] == 0 and cell[4] == 0 and cell[5] == 0:
                        startday += 1
                    cell[2] = startday
                    cell_value = datetime(*cell)
                    cell = cell_value.strftime('%Y/%m/%d %H:%M:%S')
                data.append(cell)
            evap = self.Cal_Evap(data)
            task = Cal_future(self.Cal_fore, evap)
            task.start()
            forecast = task.get_result()
            task.join()
            print(i-1, " ", data)
            print(evap)
            print(forecast)
            temp = {
                # 'Id': self.Location+1,
                'Time': data[0],
                'Temperature': float(data[1]),
                'Press': float(data[3]),
                'Wind': float(data[5]),
                'Humidity': float(data[2]),
                'Direction': float(data[4]),
            }
            if self.Len == 300:
                del(self.database[0])
                self.database.append(temp)
            else:
                self.database.append(temp)
            # 存储在数据库中的值
            test = BasicMeteo(Time=data[0], Temperature=float(data[1]), Humidity=float(data[2]), Press=float(data[3]), Direction=float(data[4]), Wind=float(data[5]))
            self.session.add(test)
            session.commit()
            print("commit:", temp)
            time.sleep(2)
        session.close()



if __name__ == "__main__":
    Session_class = sessionmaker(bind=engine) #建立与数据库的会话连接
    session = Session_class() #这里创建一个会话实例
    thread0 = Updata(filepath=args.input_filepath, session=session, model=0.1)
    thread0.start()
    thread0.join()
    # test1 = MeteoData(Id=1, Time="2021-3-26 21:11:50", Temperature=1, Humidity=1, Pressure=1, Direction=1,
    #                  Wind=1, evapration=1, future_value=1)
    # print(test1.Id)
    # session.add(test1)
    # session.commit()
    # session.close()





