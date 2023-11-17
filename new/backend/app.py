from flask import Flask, jsonify

from flask_cors import *

from api_analysis_data_view import data_view_api
from api_duct_height import duct_height_api
from api_origin_data_view import origin_data_view_api
from testapi import account_api
from api_height_predict import height_predict_api
from api_radar import radar_api

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.config['JSON_AS_ASCII'] = False  # 禁止中文转义

GLOBAL_DEBUG = True

# 注册列表
app.register_blueprint(account_api)
app.register_blueprint(data_view_api)
app.register_blueprint(origin_data_view_api)
app.register_blueprint(duct_height_api)
app.register_blueprint(height_predict_api)
app.register_blueprint(radar_api)


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/<path:fallback>')
def fallback(fallback):  # Vue Router 的 mode 为 'hash' 时可移除该方法
    if fallback.startswith('css/') or fallback.startswith('js/') \
            or fallback.startswith('img/') or fallback == 'favicon.ico':
        return app.send_static_file(fallback)
    else:
        return app.send_static_file('index.html')


def run():
    app.run(host="0.0.0.0", debug=True)


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
