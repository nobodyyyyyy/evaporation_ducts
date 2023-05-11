import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import seaborn as sns
import palettable #python颜色库


class DrawUtil:

    @staticmethod
    def draw_heatmap(arr: np.array):
        # plt.rcParams['font.sans-serif'] = ['SimHei']
        # plt.rcParams['font.sans-serif'] = ['SimHei']  # mac
        plt.rcParams['axes.unicode_minus'] = False
        df = pd.DataFrame(arr)
        p1 = sns.heatmap(df)
        plt.show()


if __name__ == '__main__':
    DrawUtil.draw_heatmap(None)