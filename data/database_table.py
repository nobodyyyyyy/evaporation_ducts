#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import argparse
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, DateTime, Float, select

parser = argparse.ArgumentParser(description='real data');
parser.add_argument('--input-filepath', type=str, default='F:\\SomeCacheFile\\Atmos_Data\\新建文件夹\\分钟数据.xlsx')
parser.add_argument('--username', type=str, default='root')
parser.add_argument('--password', type=str, default='SEU440CILab!')
parser.add_argument('--location', type=str, default='223.3.89.186')
parser.add_argument('--database', type=str, default='MeteoData')
args = parser.parse_args()

engine = create_engine("mysql+pymysql://{}:{}@{}:{}/{}".format(args.username, args.password, args.location, '3306', args.database))
Base = declarative_base()
Base.metadata.create_all(engine)
class BasicMeteo(Base):
    __tablename__ = 'basic_meteo'
    Id = Column(Integer, primary_key=True)
    Time = Column(DateTime)
    Temperature = Column(Float)
    Press = Column(Float)
    Wind = Column(Float)
    Humidity = Column(Float)
    Direction = Column(Float)

    def __repr__(self):
        bm = {
            # 'Id': self.Id,
            'times': self.Time,
            'tem': self.Temperature,
            'press': self.Press,
            'wind': self.Wind,
            'hum': self.Humidity,
            'direction': self.Direction,
        }
        return str(bm)

class EvapInfor(Base):
    __tablename__ = 'evap_infor'
    Id = Column(Integer, primary_key=True)
    Time = Column(DateTime)
    Evap_height = Column(Float)
    Futu_height = Column(Float)

    def __repr__(self):
        ei = {
            # 'Id': self.Id,
            'times': self.Time,
            'evap': self.Futu_height,
            'futu': self.Futu_height,
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
            # 'Id': self.Id,
            'times': self.Time,
            'sh': self.Surf_height,
            'ss': self.Surf_strength,
            'eb': self.Elev_bottom,
            'et': self.Elev_top,
            'es': self.Elev_strength,
        }
        return str(si)

class MeteoData(Base):
    __tablename__ = 'meteo_value'
    Id = Column(Integer, primary_key=True)
    Time = Column(DateTime)
    Temperature = Column(Float)
    Humidity = Column(Float)
    Pressure = Column(Float)
    Direction = Column(Float)
    Wind = Column(Float)
    evapration = Column(Float)
    future_value = Column(Float)

    def __repr__(self):
        meteoData = {
            'Id': self.Id,
            'tem': self.Temperature,
            'wind': self.Wind,
            'press': self.Pressure,
            'direction': self.Direction,
            'hum': self.Humidity,
            'times': self.Time,
            'evap': self.evapration,
            'futu': self.future_value
        }
        return str(meteoData)