import netCDF4 as nc
import numpy as np

from Util.TimeUtil import TimeUtil
from data.DataUtil import DataUtils

if __name__ == '__main__':

    file = 'G:\\Woof\\atmospheric\\code\\evaporation_ducts\\new\\data\\NewGen\\ncdata\\GG_nhgsmout_2021-01-09_00_00_00.nc'
    dataset = nc.Dataset(file)
    all_vars = dataset.variables.keys()

    all_vars_info = dataset.variables.items()
    all_vars_info = list(all_vars_info)
    a = np.array(dataset['g0_lon_1'])  # 72.5 ~ 135
    b = np.array(dataset['g0_lat_0'])  # 1.25 ~ 56.25

    y, x, ret = DataUtils.get_support_new_gen_data_single_date(type_='U10', lan_s=30, lan_e=100, lng_s=10, lng_e=100, file_name=file)

    print(0)
    # 新一代数值系统

     # ref Impact_height Bend_ang

    # ref = np.array(dataset.variables['Ref'][:])
    # Impact_height = np.array(dataset.variables['Impact_height'][:])
    # Bend_ang = np.array(dataset.variables['Bend_ang'][:])
    #
    # wrt = open('result.txt', mode='w')
    #
    # for _ in range(len(ref)):
    #     wrt.write('{}\t{}\t{}\n'.format(ref[_], Impact_height[_], Bend_ang[_]))

    # lat = dataset.variables['latitude'][:]
    # print(lat.shape)
    # var_data = np.array(lat)
    # print(var_data)

    # wrt = nc.Dataset('new.nc', 'w')
    # wrt.createDimension('time', len(dataset.variables['time'][:]))
    # wrt.createDimension('latitude', len(dataset.variables['lat'][:]))
    # wrt.createDimension('longitude', len(dataset.variables['lon'][:]))
    #
    # wrt.createVariable('time', 'i', ('time'))
    # wrt.createVariable('latitude', 'f', ('latitude'))
    # wrt.createVariable('longitude', 'f', ('longitude'))
    # wrt.createVariable('sst', 'f', ('time', 'latitude', 'longitude'))
    #
    # wrt.variables['time'][:] = dataset.variables['time'][:]
    # wrt.variables['latitude'][:] = dataset.variables['lat'][:]
    # wrt.variables['longitude'][:] = dataset.variables['lon'][:]
    # wrt.variables['sst'][:] = dataset.variables['sst'][:]
    #
    # wrt.close()

    # dataset.createVariable('latitude', 720)

    pass