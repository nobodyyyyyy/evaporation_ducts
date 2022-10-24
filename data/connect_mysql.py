#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, Column, Integer, DateTime, Float, String
from sqlalchemy.ext.declarative import declarative_base

# import datetime
import time
Base = declarative_base()

class MysqlOpt:
    def __init__(self, username='root', password='SEU440CILab!', address='10.201.35.215', port='3306', database='MeteoData', lim=3000):
        # mysql用户名，mysql数据库密码，数据库IP地址，数据库端口号，数据库名
        self.engine = create_engine(
            "mysql+pymysql://{}:{}@{}:{}/{}".format(username, password, address, port, database))
        # 建立与数据库的会话连接
        Base.metadata.create_all(self.engine)
        print("connect mysql successfully!")
        Session_class = sessionmaker(bind=self.engine)
        # 这里创建一个会话实例
        self.session = Session_class()
        # 读取最近的limit条数据
        self.limit = lim

        # 查询最后一个气象数据的时间

    def Query_FinalTime(self):
        result = self.session.query(BasicMeteo).order_by(BasicMeteo.Id.desc()).limit(1).all()
        if len(result) == 0:
            # 没有数据
            return '0'
        return str(result[0].Time)

        # 查询最近limit个基本气象数据

    def Query_Basic(self):
        result = self.session.query(BasicMeteo).order_by(BasicMeteo.Id.desc()).limit(self.limit).all()
        if len(result) > 0:
            result = eval(str(result))
            result = list(reversed(result))
        return result

        # 查询时间范围在start_time 到 end_time之间的数据

    def Query_Evap_Range(self, start_time, end_time):
        result = self.session.query(EvapInfor).filter(EvapInfor.Time.between(start_time, end_time)).all()
        if len(result) > 0:
            result = eval(str(result))
        return result

    def Query_Evap_init(self):
        result = self.session.query(EvapInfor).order_by(EvapInfor.Id.desc()).limit(self.limit).all()
        if len(result) > 0:
            result = eval(str(result))
            result = list(reversed(result))
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
            self.session.commit()
        except BaseException as e:
            print("basic commit error!")
            print("detail error: ", str(e))

    def Update_Evap(self, times, evap_height=0, futu_height=0, lat_lon="数据缺失"):
        newdata = EvapInfor(Time=times, Evap_height=evap_height, Futu_height=futu_height, Lat_Lon=lat_lon)
        self.session.add(newdata)
        # self.session.commit()
        try:
            self.session.commit()
        except BaseException as e:
            print("evap commit error!")
            print("detail error: ", str(e))

    def Update_SE(self, times, surf_height=0, surf_strength=0, elev_bottom=0, elev_top=0, elev_strength=0):
        newdata = SeInfor(Time=times, Surf_height=surf_height, Surf_strength=surf_strength, Elev_bottom=elev_bottom,
                          Elev_top=elev_top, Elev_strength=elev_strength)
        self.session.add(newdata)
        # self.session.commit()
        try:
            self.session.commit()
        except BaseException as e:
            print("se commit error!")
            print("detail error: ", str(e))

class BasicMeteo(Base):
    __tablename__ = 'basic_meteo'
    Id = Column(Integer, primary_key=True)
    Time = Column(DateTime)
    Temperature = Column(Float)
    Press = Column(Float)
    Wind = Column(Float)
    Humidity = Column(Float)
    Direction = Column(Float)
    SST = Column(Float)

    def __repr__(self):
        bm = {
            'times': str(self.Time),
            'tem': self.Temperature,
            'press': self.Press,
            'wind': self.Wind,
            'hum': self.Humidity,
            'direction': self.Direction,
            'sst': self.SST
        }
        return str(bm)

class EvapInfor(Base):
    __tablename__ = 'evap_infor'
    Id = Column(Integer, primary_key=True)
    Time = Column(DateTime)
    Evap_height = Column(Float)
    Futu_height = Column(Float)
    Lat_Lon = Column(String)

    def __repr__(self):
        ei = {
            'times': str(self.Time),
            'evap': self.Evap_height,
            'futu': self.Futu_height,
            'lat_lon': self.Lat_Lon
        }
        return str(ei)

class SeInfor(Base):
    __tablename__ = 'se_infor'
    Id = Column(Integer, primary_key=True)
    Time = Column(DateTime)
    Surf_height = Column(Float)
    Surf_strength = Column(Float)
    Elev_bottom = Column(Float)
    Elev_top = Column(Float)
    Elev_strength = Column(Float)

    def __repr__(self):
        si = {
            'times': str(self.Time),
            'sh': self.Surf_height,
            'ss': self.Surf_strength,
            'eb': self.Elev_bottom,
            'et': self.Elev_top,
            'es': self.Elev_strength,
        }
        return str(si)

