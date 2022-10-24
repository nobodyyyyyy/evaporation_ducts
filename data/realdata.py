#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import xlrd
import sys
import pandas
from xlrd import xldate_as_tuple
import matlab.engine
import argparse

from datetime import datetime, timedelta
from sqlalchemy import create_engine

parser=argparse.ArgumentParser(description='real data');
parser.add_argument('--input-filepath', type=str, default='F:\\SomeCacheFile\\Atmos_Data\\新建文件夹\\分钟数据.xlsx')
parser.add_argument('--username', type=str, default='root')
parser.add_argument('--password', type=str, default='SEU440CILab!')
parser.add_argument('--location', type=str, default='223.3.89.186')
parser.add_argument('--database', type=str, default='MeteoData')
args = parser.parse_args()


# engine=create_engine("mysql+pymysql://{}:{}@{}:{}/{}".format(args.username, args.password, args.location, '3306', args.database))
# sql_query='insert from meteo_value'
# pd.read_sql_query(sql_query, engine)
# if __name__=='__main__':
    # print(args.input_filepath)
    # try:
    #     ws=xlrd.open_workbook(args.input_filepath)
    # except:
    #     print('cannot find this file！')
    # wp=ws.sheet_by_index(0)
    # print('name', wp.name)
    # print('cols', wp.ncols)
    # print('rows', wp.nrows)
    # nrows=wp.nrows
    # ncols=wp.ncols
    # startday=1;
    # for i in range(nrows):
    #     data = []
    #     if i < 2:
    #         continue
    #     for j in range(ncols):
    #         cell = wp.cell_value(i, j)
    #         if j==0:
    #             cell=list(xldate_as_tuple(cell, 0))
    #             cell[0]=2021
    #             cell[1]=3
    #             if cell[3]==0 and cell[4]==0 and cell[5]==0:
    #                 startday+=1
    #             cell[2]=startday
    #             cell_value=datetime(*cell)
    #             cell=cell_value.strftime('%Y/%m/%d %H:%M:%S')
    #         data.append(cell)
    #     print(data)

