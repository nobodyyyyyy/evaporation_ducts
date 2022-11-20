import netCDF4 as nc
import numpy as np

if __name__ == '__main__':

    file = './CN/ERA5_hourly_00_sst_2021.nc'
    dataset = nc.Dataset(file)
    all_vars = dataset.variables.keys()

    all_vars_info = dataset.variables.items()
    all_vars_info = list(all_vars_info)
    a = dataset.variables['sst'][:]
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