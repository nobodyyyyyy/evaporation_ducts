from flask import Flask

from flask_cors import *

from new.flask.api_analysis_data_view import data_view_api
from new.flask.api_duct_height import duct_height_api
from new.flask.api_origin_data_view import origin_data_view_api
from new.flask.testapi import account_api
from new.flask.api_height_predict import height_predict_api
from new.flask.api_radar import radar_api

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


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
