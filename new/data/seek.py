import netCDF4 as nc
import numpy as np

if __name__ == '__main__':

    file = './AEM/omega.2018-01.daily.nc'
    dataset = nc.Dataset(file)
    all_vars = dataset.variables.keys()

    all_vars_info = dataset.variables.items()
    all_vars_info = list(all_vars_info)
    lat = dataset.variables['latitude'][:]
    print(lat.shape)
    var_data = np.array(lat)
    print(var_data)
