#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from flask import Flask, request, url_for, flash, session
from flask_wtf import Form
from wtforms import StringField
from wtforms.validators import DataRequired
from threading import Lock
from data.generation import *
from data.algorithm import xuankong, biaomian, dianciLoss, getTrap
from data.radar import *
from flask import render_template, redirect, abort
from flask_socketio import SocketIO, emit
from datetime import *
from data.connect_mysql import MysqlOpt
from data.FileHelper import dataReader
from data.StoreInfor import DuctInfor

#船上数据库名
db3 = 'hxywq.db3'
conn = sqlite3.connect(db3)
c = conn.cursor()
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
limit_num = 3000
socketio = SocketIO(app)
# 缓存1秒，用来解决静态文件不刷新
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)
thread = None
thread_lock = Lock()
# 可以操作数据库的对象
Q = MysqlOpt()
# 可以读取文件的Generation，为检测bug，暂时注释掉
#初始化online method
evap_arr = []
for i in range(100):
    evap_arr.append(random.random() * 50)
OM = Online_Method(evap_arr)
old_gru = torch.load('model\\old_gru.pt')
old_lstm = torch.load('model\\old_lstm.pt')
old_tcn = torch.load('model\\old_tcn.pt')
old_svr = joblib.load('model\\old_svr.model')
old_rf = joblib.load('model\\old_rf.model')
OM.Models = [old_gru, old_tcn, old_lstm, old_svr, old_rf]
OM.setMax_Min()
G = Generation(sqlconnect=Q, OM=OM, socketio=socketio, cursor=c, limit=limit_num)
dataHelper = dataReader()
ductStore = DuctInfor()
print('----G----')
R = Radar_Coe()
print("雷达参数结束")

class Radar_information(Form):
    # 雷达频率
    RF = StringField('RF')
    # 雷达峰值频率
    RT = StringField('RT')
    # 天线高度
    AH = StringField('AH')
    # 天线增益
    AG = StringField('AG')
    # 波束宽度
    BW = StringField('BW')
    # 发射仰角
    LE = StringField('LE')
    # 最小信噪比
    MN = StringField('MN')
    # 接收机带宽
    RW = StringField('RW')
    # 系统综合损耗
    SL = StringField('SL')
    # 接收机噪声系数
    NC = StringField('NC')
    # 目标高度
    TH = StringField('TH')
    # 目标散射截面
    RR = StringField('RR')

@app.route('/')
@app.route('/index')
def index():
    data_num = len(G.database)
    if data_num < 300:
        return render_template('index.html', Data=G.database)
    else:
        Data = G.database[data_num-300:]
        return render_template('index.html', Data=Data)

# 上传文件(func)
@app.route('/uploadS', methods=["POST"])
def upload_surf():
    fileContent = request.files['file-data'].read()
    fileName = request.files.get('file-data').filename
    tempType = fileName.split(".")[-1]
    # 错误码
    error = {
        "infor": "OK",
        "code": 200
    }
    # 这里要返回ref，h，以及表面，悬空波导高度信息
    # 并保存ref，h的信息，以供电磁波使用
    if tempType == "txt":
        try:
            dataHelper.readTxt(fileContent, fileName)
        except BaseException as err:
            error['infor'] = err
            error['code'] = 400
            return render_template('surfaceevaporation.html', Refraction=None, SD=None,
                                   ED=None, FileName=fileName, ErrorCode=error)
    elif tempType == "tpu":
        try:
            dataHelper.readTpu(fileContent, fileName)
        except BaseException as err:
            error['infor'] = err
            error['code'] = 500
            return render_template('surfaceevaporation.html', Refraction=None, SD=None,
                                   ED=None, FileName=fileName, ErrorCode=error)
    elif tempType != "txt" and tempType != "tpu":
        error['infor'] = "bad file type"
        error['code'] = 300
        return render_template('surfaceevaporation.html', Refraction=None, SD=None,
                                   ED=None, FileName=fileName, ErrorCode=error)
    # 计算大气折射率
    ref, h = G.generate_interpolation_data(dataHelper.dataset)
    ductStore.ref = ref
    ductStore.height = h
    # 计算表面波导和悬空波导
    elevated_duct, surface_duct = xuankong(ref, h), biaomian(ref, h)
    trapFreq, cutOff = getTrap(elevated_duct, surface_duct)
    if elevated_duct[0] != 0:
        elevated_duct = elevated_duct[4:7].flatten().tolist()
        elevated_duct[0], elevated_duct[1] = elevated_duct[1], elevated_duct[0]
        surface_duct = [0, 0, 0]
    elif surface_duct[0] != 0:
        elevated_duct = [0, 0, 0]
        surface_duct = surface_duct[3:5].flatten().tolist()
        surface_duct.insert(0, 0)
    else:
        elevated_duct = [0, 0, 0]
        surface_duct = [0, 0, 0]
    Electron = []
    loss = dianciLoss(ref, h)
    for i in range(200):
        for j in range(200):
            if loss is None:
                Electron.append([i, j, random.randint(0, 200)])
            else:
                Electron.append([i, j, np.float(loss[i][j])])
    # 高度单位换算
    h = h * 1000
    elevated_duct[0] *= 1000
    elevated_duct[1] *= 1000
    surface_duct[0] *= 1000
    surface_duct[1] *= 1000
    refraction = np.append(ref, h, axis=1).tolist()
    # 存储
    ductStore.surfaceDuct = surface_duct
    ductStore.elevatedDuct = elevated_duct
    ductStore.trapInfor['trapFreq'] = trapFreq
    ductStore.trapInfor['cutOff'] = cutOff
    ductStore.electron = Electron
    return render_template('surfaceevaporation.html', Refraction=refraction, SD=ductStore.surfaceDuct,
                           ED=ductStore.elevatedDuct, FileName=fileName, ErrorCode=error, TrapInfor=ductStore.trapInfor)

@app.route('/uploadE', methods=["POST"])
def upload_elec():
    error = {
        "infor": "OK",
        "code": 200
    }
    # 雷达计算
    Electron = []
    UpData = R.get()
    Result = R.LossFlag
    form = Radar_information()
    # 判断是否验证提交
    if form.validate_on_submit():
        # 雷达频率(MHz)
        RF = form.RF.data
        # 雷达峰值功率(KW)
        RT = form.RT.data
        # 天线高度(m)
        AH = form.AH.data
        # 天线增益(dB)
        AG = form.AG.data
        # 波束宽度
        BW = form.BW.data
        # 发射仰角
        LE = form.LE.data
        # 最小信噪比
        MN = form.MN.data
        # 接收机带宽
        RW = form.RW.data
        # 系统综合损耗
        SL = form.SL.data
        # 接收机噪声系数
        NC = form.NC.data
        # 目标高度
        TH = form.TH.data
        # 目标散射截面
        RR = form.RR.data
        R.updata(RF, RT, AH, AG, BW, LE, MN, RW, SL, NC, TH, RR)
        UpData = R.get()
        Result = R.LossFlag
        # 计算大气折射率
        ref, h = None, None
        if ductStore.ref is not None:
            ref, h = ductStore.ref, ductStore.height
        elif dataHelper.filename is not None:
            ref, h = G.generate_interpolation_data(dataHelper.dataset)
        # 电磁波传播损耗
        if ductStore.electron is not None:
            pass
        elif ref is not None:
            loss = dianciLoss(ref, h)
            for i in range(200):
                for j in range(200):
                    if loss is None:
                        Electron.append([i, j, random.randint(0, 200)])
                    else:
                        Electron.append([i, j, np.float(loss[i][j])])
            ductStore.electron = Electron
        return render_template('electromagenetic.html', Elec=Electron, FileName=dataHelper.filename, R_I=UpData,
                               form=form, R_R=Result, ErrorCode=error, TrapInfor=ductStore.trapInfor)

    fileName = request.files.get('file-data').filename
    fileContent = request.files['file-data'].read()
    tempType = fileName.split(".")[-1]
    # 并保存ref，h的信息，以供电磁波使用
    if tempType == "txt":
        try:
            dataHelper.readTxt(fileContent, fileName)
        except BaseException as err:
            error['infor'] = err
            error['code'] = 400
            return render_template('electromagenetic.html', Elec=Electron, FileName=dataHelper.filename, R_I=UpData,
                                   form=form, R_R=Result, ErrorCode=error, TrapInfor=ductStore.trapInfor)
    elif tempType == "tpu":
        try:
            dataHelper.readTpu(fileContent, fileName)
        except BaseException as err:
            error['infor'] = err
            error['code'] = 500
            return render_template('electromagenetic.html', Elec=Electron, FileName=dataHelper.filename, R_I=UpData,
                                   form=form, R_R=Result, ErrorCode=error, TrapInfor=ductStore.trapInfor)
    elif tempType != "txt" and tempType != "tpu":
        error['infor'] = "bad file type"
        error['code'] = 300
        return render_template('electromagenetic.html', Elec=Electron, FileName=dataHelper.filename, R_I=UpData,
                               form=form, R_R=Result, ErrorCode=error, TrapInfor=ductStore.trapInfor)
    # 计算大气折射率
    ref, h = G.generate_interpolation_data(dataHelper.dataset)
    elevated_duct, surface_duct = xuankong(ref, h), biaomian(ref, h)
    ductStore.trapInfor['trapFreq'], ductStore.trapInfor['cutOff'] = getTrap(elevated_duct, surface_duct)
    loss = dianciLoss(ref, h)
    if elevated_duct[0] != 0:
        elevated_duct = elevated_duct[4:7].flatten().tolist()
        elevated_duct[0], elevated_duct[1] = elevated_duct[1], elevated_duct[0]
        surface_duct = [0, 0, 0]
    elif surface_duct[0] != 0:
        elevated_duct = [0, 0, 0]
        surface_duct = surface_duct[3:5].flatten().tolist()
        surface_duct.insert(0, 0)
    else:
        elevated_duct = [0, 0, 0]
        surface_duct = [0, 0, 0]
    ductStore.elevatedDuct = elevated_duct
    ductStore.surfaceDuct = surface_duct
    for i in range(200):
        for j in range(200):
            if loss is None:
                Electron.append([i, j, random.randint(0, 200)])
            else:
                Electron.append([i, j, np.float(loss[i][j])])
    ductStore.height, ductStore.ref = h, ref
    ductStore.electron = Electron
    return render_template('electromagenetic.html', Elec=ductStore.electron, FileName=dataHelper.filename, R_I=UpData,
                           form=form, R_R=Result, ErrorCode=error, TrapInfor=ductStore.trapInfor)
@app.route('/evaporation')
def evaporation_page():
    data_num = len(G.database)
    # 每次仅需要选取最近的气象信息
    # 不需要每次都传输所有的历史气象数据
    BM = G.database[-1]
    if data_num < 300:
        return render_template('evaporation.html', MeteoData=BM, Evap=G.evap_duct, Request=None, RequestTime=None)
    else:
        return render_template('evaporation.html', MeteoData=BM, Evap=G.evap_duct, Request=None, RequestTime=None)

@app.route('/history', methods=['GET', 'POST'])
def history_evap():
    # data_num = len(G.database)
    # 每次仅需要选取最近的气象信息
    # 不需要每次都传输所有的历史气象数据
    BM = G.database[-1]
    if request.method == 'POST':
        start = request.form.get('starttime') + " 00:00:00"
        end = request.form.get('endtime') + " 23:59:59"
        start = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
        end = datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
        print("查询历史波导信息时间区间：")
        print(start, end)
        request_time = {
            'start_time':start,
            'end_time':end
        }
        # 获取时间，通过时间查找历史波导信息
        # ----------------------------------------------------------------------------------
        request_data = Q.Query_Evap_Range(start, end)
        # ----------------------------------------------------------------------------------
        # 以上的代码将在数据库构建完成的时候，换成使用start和end对数据库中的数据进行查找
        return render_template('evaporation.html', MeteoData=BM, Evap=G.evap_duct, Request=request_data, RequestTime=request_time)
    return render_template('evaporation.html', MeteoData=BM, Evap=G.evap_duct, Request=None, RequestTime=None)

@app.route('/futureheight')
def futureheight_page():
    # 每次仅需要选取最近的气象信息
    # 不需要每次都传输所有的历史气象数据
    BM = G.database[-1]
    return render_template('futureheight.html', MeteoData=BM, Evap=G.evap_duct)

@app.route('/electromagenetic', methods=['POST', 'GET'])
def electro_page():
    error = {
        "infor": "OK",
        "code": 200
    }
    form = Radar_information()
    # 判断是否验证提交
    if form.validate_on_submit():
        # 雷达频率(MHz)
        print("electro validate!")
        RF = form.RF.data
        # 雷达峰值功率(KW)
        RT = form.RT.data
        # 天线高度(m)
        AH = form.AH.data
        # 天线增益(dB)
        AG = form.AG.data
        # 波束宽度
        BW = form.BW.data
        # 发射仰角
        LE = form.LE.data
        # 最小信噪比
        MN = form.MN.data
        # 接收机带宽
        RW = form.RW.data
        # 系统综合损耗
        SL = form.SL.data
        # 接收机噪声系数
        NC = form.NC.data
        # 目标高度
        TH = form.TH.data
        # 目标散射截面
        RR = form.RR.data
        R.updata(RF, RT, AH, AG, BW, LE, MN, RW, SL, NC, TH, RR)
    UpData = R.get()
    # R_I 为现有的雷达相关系数
    # R_R 是雷达经过计算后得到的结果，为一个数值
    # 当雷达的系数R_I不变的时候，R_R为一个固定值，所以不需要实时计算传输
    Result = R.LossFlag
    return render_template('electromagenetic.html', Elec=ductStore.electron, FileName=dataHelper.filename, R_I=UpData,
                           form=form, R_R=Result, ErrorCode=error, TrapInfor=ductStore.trapInfor)

@app.route('/surface-evap')
def surface_page():
    error = {
        "infor": "OK",
        "code": 200
    }
    if dataHelper.filename is not None:
        # 计算大气折射率
        ref, h = ductStore.ref, ductStore.height
        # 高度单位换算
        h = h * 1000
        refraction = np.append(ref, h, axis=1).tolist()
        return render_template('surfaceevaporation.html', Refraction=refraction, SD=ductStore.surfaceDuct,
                               ED=ductStore.elevatedDuct, FileName=dataHelper.filename, ErrorCode=error, TrapInfor=ductStore.trapInfor)
    return render_template('surfaceevaporation.html', Refraction=None, SD=None, ED=None, FileName=None, ErrorCode=error, TrapInfor=ductStore.trapInfor)


@socketio.on('connect', namespace='/test_conn')
def transfer_infor():
    print("开始后台更新")
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(background_thread)

# 最后的逻辑是读取气象文件
# 将气象文件中的数据获取后即可以进行相应的操作
# 将socketio的相关内容编写在后台成立数据的线程的数据中
def background_thread():
    while True:
        print('--------------------')
        G.update()
        socketio.sleep(5)
        print('--------------------')

if __name__ == '__main__':
    app.jinja_env.auto_reload = True
    socketio.run(app, debug=False)


