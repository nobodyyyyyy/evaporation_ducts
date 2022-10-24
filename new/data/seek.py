import netCDF4 as nc
import numpy as np

if __name__ == '__main__':

    file = './AEM/omega.2018-01.daily.nc'
    dataset = nc.Dataset(file)
    all_vars = dataset.variables.keys()
    # print(len(all_vars))
    # 获取所有变量信息
    all_vars_info = dataset.variables.items()
    all_vars_info = list(all_vars_info)
    # print(all_vars_info)
    # 获取单独的一个变量的数据
    precipitationCal = dataset.variables['level'][:]
    print(precipitationCal.shape)
    # 转换成数组
    var_data = np.array(precipitationCal)
    print(var_data)
