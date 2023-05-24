# Data folder structure:
# -AIS
#   |-2017
#       |-Azteca 1
#           |-A1-V1-2017.csv
#           |-A1-V2-2017.csv
#           |-A1-V3-2017.csv
#           |-A1-V4-2017.csv
#       |-Azteca 2
#       .
#       .
#       .
#   |-bathymetry
#       |-ETOPO1_Bed_g_gmt4.grd
#   |-Globocolor
#       |-2017
#           |-01 (month)
#               |-01 (day)
#                   |-L3m_20170101__GLOB_25_GSM-MODVIR_CHL1_DAY_00.nc
#       |-2018
#       .
#       .
#       .
#   |-SST
#       |-2017
#           |-01 (month)
#               |-01 (day)
#                   |-20170101120000-NCEI-L4_GHRSST-SSTblend-AVHRR_OI-GLOB-v02.0-fv02.1.nc
#       |-2018
#       .
#       .
#       .
#   |-SLA
#       |-dt_global_allsat_phy_l4_20170101_20210318.nc
#       |-dt_global_allsat_phy_l4_20170102_20210318.nc
#       |-dt_global_allsat_phy_l4_20170103_20210318.nc
#       .
#       .
#       .
import glob
import os
from datetime import datetime
from math import sin, asin, sqrt, radians, cos

import netCDF4 as nc
import xarray as xr
import numpy as np
import pandas
import pandas as pd
from scipy.spatial import KDTree

from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors, KDTree
from utils.config_loader import Configs
import regex as re


class DataLoader:
    def __init__(self, year_range):
        self.df = None
        self.data_dir = Configs.get('DATA_DIR')
        self.year_range = year_range
        if len(year_range) == 1:
            self.year_name = str(year_range[0])
        elif len(year_range) == 2:
            self.year_name = str(year_range[0]) + '_' + str(year_range[1])
        self.df_path = os.path.join(self.data_dir, 'catch_' + self.year_name + '_final.ftr')

        self.relevant_data_variables = ['year', 'month', 'lat_ref', 'lon_ref',
                                        'total_catch', 'total_effort',
                                        'chl1_mean', 'sst', 'sla', 'eke', 'z']

    def load_data(self):
        if os.path.exists(self.df_path):
            self.df = pd.read_feather(self.df_path)
            self.preprocess()
            return self.df
        else:
            raise Exception('Data file not found. Please run data_preprocessing.py first.')

    def preprocess(self):
        # remove any rows with NaN values
        self.df = self.df.dropna()
        # drop the rows for which any column with "mask" in the name is True
        self.df = self.df[~self.df.filter(regex='mask').any(axis=1)]
        # go th
        # extract effort data from the comments column
        self.df['total_effort'] = self.df['comments'].apply(self.extract_effort)
        # drop the rows with NaN values
        self.df = self.df.dropna()

        self.df['year'] = self.df['date'].apply(lambda x: int(x.split('-')[0]))
        self.df['month'] = self.df['date'].apply(lambda x: int(x.split('-')[1]))

        # convert total_catch to integer
        self.df['total_catch'] = self.df['total_catch'].astype(int)

        # keep the relevant columns
        self.df = self.df[self.relevant_data_variables]
        self.df.reset_index(drop=True, inplace=True)




    def extract_effort(self, comment):
        effort = np.nan
        if comment is not None:
            # extract numbers with in the comment string
            numbers = re.findall(r'\d+', comment)
            # convert the numbers to integers and add them
            effort = sum([int(n) for n in numbers])
        return effort

if __name__ == '__main__':
    dl = DataLoader([2017])
    dl.load_data()
