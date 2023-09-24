import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import seaborn as sns
# import palettable #python颜色库


class DrawUtil:

    @staticmethod
    def draw_l_single_heatmap(arr: np.array, square_ticking=True, title=''):
        # plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rc('font', family='SimHei', weight='bold')
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(12, 5))
        plt.title(title)
        if square_ticking:
            x_len = arr.shape[0]
            y_len = arr.shape[1]
            x_tick = np.arange(0, x_len)
            y_tick = np.arange(0, y_len)
            df = pd.DataFrame(arr, index=x_tick, columns=y_tick)
        else:
            df = pd.DataFrame(arr)
        p1 = sns.heatmap(df, cmap='RdYlBu_r', vmax=220).invert_yaxis()
        plt.xlabel('范围/km')
        plt.ylabel('高度/m')
        plt.show()

    @staticmethod
    def draw_line_chart(x_axis_data, y_axis_data):

        plt.plot(x_axis_data, y_axis_data, 'r', alpha=0.5, linewidth=1)

        plt.legend()  # 显示上面的label
        plt.xlabel('height(m)')  # x_label
        plt.ylabel('distance(m)')  # y_label

        # plt.ylim(-1,1)#仅设置y轴坐标范围
        plt.show()


if __name__ == '__main__':
    pass