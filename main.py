import netCDF4 as nc
import os
from utils.config_loader import Configs

DATA_DIR = Configs.get('DATA_DIR')

chl_dir = os.path.join(DATA_DIR, 'Globocolor')

parent_dir = r'K:\oceanography\Data\Globocolor\daily\2017\06\05'

file_name = 'L3m_20170605__GLOB_25_GSM-MODVIR_CHL1_DAY_00.nc'

file_path = os.path.join(parent_dir, file_name)

dt = nc.Dataset(file_path)

# list the dt variable names

vars = dt.variables.keys()

# lat, lon, chl1_mean, chl1_flags, chl1_error

# for root, dirs, files in os.walk(chl_dir, topdown=True, onerror=None, followlinks=False):
#     print('root:', root)
#     print('dirs:', dirs)
#     print('files:', files)
#     if not len(files):
#         continue





