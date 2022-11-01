import time


class TimeUtil:

    @staticmethod
    def format_month_or_day(input_):
        """
        将月或日前置 0
        :return:
        """
        if isinstance(input_, str):
            if len(input_) == 1:
                return '0' + input_
            else:
                print('input_ {} is str'.format(input_))
                return input_
        elif isinstance(input_, int):
            if input_ > 10:
                return str(input_)
            elif input_ > 0:
                return '0' + str(input_)
        print('cannot transform for input_ {}'.format(input_))
        return input_


    @staticmethod
    def current_time_millis():
        return round(time.time() * 1000)


    @staticmethod
    def time_millis_2_nc_timestamp(time_millis):
        """
        nc 文件中的时间是 hours since 1900-01-01 00:00:00.0
        Unix 时间戳是从1970年1月1日（UTC/GMT的午夜）开始所经过的秒数
        转换 Unix 时间戳（秒）至 nc 时间戳
        :param time_millis: Unix 时间戳
        :return: nc 时间戳
        """
        day_diff = 25567
        sec_diff = day_diff * 24 * 3600
        return int((time_millis + sec_diff) / 3600)


    @staticmethod
    def nc_timestamp_2_time_millis(nc_timestamp):
        """
        nc 时间戳转 Unix 时间戳（秒）
        """
        day_diff = 25567
        sec_diff = day_diff * 24 * 3600
        return nc_timestamp * 3600 - sec_diff


    @staticmethod
    def get_day_sum(year, month):
        if month in (1, 3, 5, 7, 8, 10, 12):
            return 31
        elif month in (4, 6, 9, 11):
            return 30
        elif month == 2:
            if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
                return 29
            else:
                return 28
