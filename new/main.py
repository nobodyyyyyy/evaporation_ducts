import os.path
import sys
sys.path.append(os.path.dirname(os.getcwd()))

# 路径加入
sys.path.append('./backend')
sys.path.append('./data')
sys.path.append('./height_model')
sys.path.append('./height_predict')
sys.path.append('./radar_model')
sys.path.append('./Util')


from flask import Flask
from flask_cors import *
from gevent import pywsgi

from backend.api_analysis_data_view import data_view_api
from backend.api_duct_height import duct_height_api
from backend.api_origin_data_view import origin_data_view_api
from backend.testapi import account_api
from backend.api_height_predict import height_predict_api
from backend.api_radar import radar_api

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.config['JSON_AS_ASCII'] = False  # 禁止中文转义

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


if __name__ == '__main__':
    print('处理中……')

    # server = pywsgi.WSGIServer(('127.0.0.1', 5000), app)
    # print('请在浏览器中输入网址: http://localhost:5000/login')
    # server.serve_forever()

    app.run(host="0.0.0.0", debug=True)

