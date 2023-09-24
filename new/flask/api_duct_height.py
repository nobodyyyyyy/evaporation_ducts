from flask import Blueprint, request
import json

from new.flask.SupportedSingletons import SupportedSingletons
from new.height_model.models.babin import babin_duct_height
from new.height_model.models.nps import nps_duct_height
from new.height_model.models.pj2height import pj2height
from new.height_model.models.sst import evap_duct_SST

duct_height_api = Blueprint('ductHeight', __name__)

prefix = '/height/'

API_CAL_HEIGHT = prefix + '/cal'

# 单例列表
modules = SupportedSingletons()

model_entry = {
    'nps': 0,
    'babin': 1,
    'liuli': 2,
    'pj': 3
}
models = [nps_duct_height, babin_duct_height, evap_duct_SST, pj2height]


@duct_height_api.route(API_CAL_HEIGHT)
def cal_height():
    data = request.get_json()
    print(data)
    t = data['temp']  # 气温
    eh = data['relh']  # 相对湿度
    u = data['speed']  # 风速
    p = data['pressure']  # 压强
    h = data['height']  # 测量高度
    sst = data['sst']  # 海面温度
    model_name = data['model']
    model = models[model_entry[model_name]]
    try:
        res = model(t, eh, sst, u, p, h)
    except Exception as e:
        print('cal_height... Error: {}'.format(e))
        return json.dumps({
            'code': -1,
            'msg': e
        })
    return json.dumps({
        'code': 0,
        'res': res
    })
