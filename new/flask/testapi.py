from flask import Blueprint

account_api = Blueprint('account_api', __name__)


@account_api.route("/account")
def account_list():
    return "list of accounts"