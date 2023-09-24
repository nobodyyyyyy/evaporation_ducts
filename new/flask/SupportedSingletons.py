import pandas as pd

from new.data.DataUtil import DataUtils
from new.height_model.HeightCal import HeightCal
from new.height_predict.entry import PredictModel
from new.radar_model.RadarCal import RadarCal


class SupportedSingletons:
    """
    存所有单例
    """

    def __init__(self):
        self.heightCal = HeightCal('../data/sounding')
        self.dataUtil = DataUtils()
        self.heightPredict = PredictModel()
        self.radarCal = RadarCal()

    def height_cal(self):
        return self.heightCal

    def data_util(self):
        return self.dataUtil

    def height_predict(self):
        return self.heightPredict

    def radar_cal(self):
        return self.radarCal


if __name__ == '__main__':
    # a = SupportedSingletons()
    # b = SupportedSingletons()
    print(1)

    a = pd.date_range(1695004304, 1696004304)
    print(2)

# a[2].strftime('%Y-%m-%d')