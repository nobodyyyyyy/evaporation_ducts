import os

import pandas as pd

from new.data.DataUtil import DataUtils


def merge_real_heights_and_model_heights(real_heights_root, model_heights_root, dest_root):

    reals = DataUtils.get_all_file_names(real_heights_root)
    models = DataUtils.get_all_file_names(model_heights_root)
    os.makedirs(dest_root, exist_ok=True)

    for model_dir in models:
        if model_dir in reals:
            _real_dir = real_heights_root + model_dir
            _model_dir = model_heights_root + model_dir
            _model_data = pd.read_excel(_model_dir)
            _real_data = pd.read_excel(_real_dir)

            _reals = _real_data.iloc[:,1]
            _reals = pd.DataFrame({'真实高度【伪折射率模型计算结果】': _reals.tolist()})
            _model_data = pd.concat([_model_data, _reals], axis=1)

            _model_data.to_excel(dest_root + model_dir, index=False)

if __name__ == '__main__':
    merge_real_heights_and_model_heights('./real_heights/', './selected_stations/', './merged/')