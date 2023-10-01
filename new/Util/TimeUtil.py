import time
import datetime


class TimeUtil:

    @staticmethod
    def format_month_or_day(input_):
        """
        将月或日前置 0
        :return:
        """
        if isinstance(input_, str):
            input_ = int(input_)
        if isinstance(input_, int):
            if input_ >= 10:
                return str(input_)
            elif input_ > 0:
                return '0' + str(input_)
        print('cannot transform for input_ {}'.format(input_))
        return input_

    @staticmethod
    def format_date_to_year_month_day(str_like_date, format_="%Y-%m-%d"):
        """
        '2022-01-02' -> 2022, 1, 2
        """
        t = time.strptime(str_like_date, format_)
        return t.tm_year, t.tm_mon, t.tm_mday

    @staticmethod
    def current_time_millis():
        return round(time.time() * 1000)

    @staticmethod
    def current_time_str(format_='%Y-%m-%d %H:%M:%S'):
        ret = time.strftime(format_, time.localtime(int(time.time())))
        return ret

    @staticmethod
    def to_time_millis(year_, month_, day_, hr_, min_, sec_):
        dt = datetime.datetime(year_, month_, day_, hr_, min_, sec_)
        return int(time.mktime(dt.timetuple()))

    @staticmethod
    def timestamp_to_datetime(ts, format_="%Y-%m-%d"):
        from datetime import datetime
        tmp = datetime.strptime(time.strftime(format_, time.localtime(ts)), format_)
        return datetime.date(tmp)

    @staticmethod
    def str_to_timestamp(date_str, format_="%Y-%m-%d"):
        return int(time.mktime(time.strptime(date_str, format_))) * 1000

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
    def noaa_timestamp_2_time_millis(number_of_delta_t):
        """
        noaa 的 timestamp 表示的是与 1800.1.1 相差的天数
        """
        begin = datetime.datetime(1800, 1, 1, 0, 0, 0)
        ret = begin + datetime.timedelta(days=number_of_delta_t)
        return int(time.mktime(ret.timetuple()))

    @staticmethod
    def time_millis_2_noaa_timestamp(time_millis):
        begin = datetime.datetime(1800, 1, 1, 0, 0, 0)
        cur = datetime.datetime.fromtimestamp(time_millis)
        ret = cur - begin
        return ret.days

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


if __name__ == '__main__':
    print(TimeUtil.time_millis_2_noaa_timestamp(1609344000))
