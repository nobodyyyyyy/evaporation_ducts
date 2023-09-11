from flask import Blueprint

data_view_api = Blueprint('dataview', __name__)

prefix = 'dataview/'

@data_view_api.route("/account")
def account_list():
    return "list of accounts"