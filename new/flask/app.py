from flask import Flask, jsonify, request

from new.flask.SupportedSingletons import SupportedSingletons
from testapi import account_api
from api_analysis_data_view import data_view_api
from api_origin_data_view import origin_data_view_api
from api_duct_height import duct_height_api

from flask_cors import *

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.config['JSON_AS_ASCII'] = False  # 禁止中文转义

# 注册列表
app.register_blueprint(account_api)
app.register_blueprint(data_view_api)
app.register_blueprint(origin_data_view_api)
app.register_blueprint(duct_height_api)

# 单例列表
modules = SupportedSingletons()


@app.route('/login', methods=['POST', 'GET'])
def returns():
    datas = {'name': '张三', 'age': 18}
    return jsonify(datas)


@app.route("/user/login", methods=["POST"])
def user_login():
    """
    用户登录
    :return:
    """
    data = request.get_json()
    userName = data.get("userName")
    password = data.get("password")
    if userName == "admin" and password == "123456":
        return jsonify({
            "code": 0,
            "data": {
                "token": "666666"
            }
        })
    else:
        return jsonify({
            "code": 99999999,
            "msg": "用户名或密码错误"
        })


@app.route("/user/info", methods=["GET", "POST"])
def user_info():
    """
    获取当前用户信息
    :return:
    """
    token = request.headers.get("token")
    if token == "666666":
        return jsonify({
            "code": 0,
            "data": {
                "id": "1",
                "userName": "admin",
                "realName": "张三",
                "userType": 1
            }
        })
    return jsonify({
        "code": 99990403,
        "msg": "token不存在或已过期"
    })


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
