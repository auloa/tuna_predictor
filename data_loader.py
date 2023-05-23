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


class DataLoader:
    def __init__(self, year_range, testing=False):
        self.data_dir = Configs.get('DATA_DIR')
        self.year_range = year_range
        self.testing = testing

    def load_data(self):
        # read catch data
        # list file names with "catch" in the name
        catch_files = glob.glob(os.path.join(self.data_dir, 'catch*.ftr'))
        # keep the files that are in the year range
        catch_files = [f for f in catch_files if int(f.split('_')[-1].split('.')[0]) in self.year_range]
        # read all files into a list of dataframes
        df_catch = [pd.read_feather(f) for f in catch_files]
        # concatenate all dataframes into one
        df_catch = pd.concat(df_catch, ignore_index=True)

        # go through each date and load the relevant data from sst, sla, chl, and bathymetry

        # get the relevant dates
        dates_ = df_catch['date'].unique()
        # # convert dates_ to datetime objects
        # # group the dates by year
        # dates = {}
        # for year in self.year_range:
        #     dates[datetime.datetime(str(year))] = [datetime.strptime(d, '%d-%m-%y') for d in dates_ if d.year == datetime(str(year), 1, 1)]

        year_month = []
        year_month_day = []
        # get the list of unique year month combinations in the dates
        for date in dates_:
            # get the year and month in string format
            date = datetime.strptime(date, "%d-%m-%y")
            year = str(date.year)
            month = str(date.month)
            day = str(date.day).zfill(2)
            year_month.append(year + '_' + month)
            year_month_day.append(year + '-' + month.zfill(2) + '-' + day)
        year_month = list(set(year_month))

        # load the data for each year month combination
        # list all files with the year month combination in the name

        list_sst = []
        list_chl = []
        list_sla = []
        list_bath = []
        for year_month in tqdm(year_month):
            # load sst
            list_sst.append(os.path.join(self.data_dir, f'sst_{year_month}.ftr'))
            # load chl
            list_chl.extend(glob.glob(os.path.join(self.data_dir, f'chl_{year_month}.ftr')))
            # load sla
            list_sla.extend(glob.glob(os.path.join(self.data_dir, f'sla_{year_month}.ftr')))
            # load bathymetry
        bath_file = os.path.join(self.data_dir, f'bathymetry.ftr')
        # load bathymetry
        df_bath = pd.read_feather(bath_file)

        # match the bathymetry to the catch data using the lat_ref and lon_ref columns while only taking column "z"
        # from the bathymetry
        df_catch = pd.merge(df_catch, df_bath[['lat_ref', 'lon_ref', 'z']], on=['lat_ref', 'lon_ref'], how='left')

        del (df_bath)

        # list the data variables to use
        data_vars = ['date', 'sst', 'sst_mask', 'chl1_mean', 'chl1_mean_mask', 'sla', 'ekel', 'z', 'lat_ref', 'lon_ref']
        # list the dataframes to use
        # start loading data from one dataset at a time
        # load sst
        df = pd.DataFrame()
        df_sst = pd.DataFrame()
        for f in tqdm(list_sst, desc='Loading SST'):
            # load the data
            if not os.path.exists(f):
                continue
            df_l = pd.read_feather(f)
            # keep the data belonging to the dates in year_month_day
            df_l = df_l[df_l['date'].isin(year_month_day) & df_l.sst_mask == False][
                ['date', 'sst', 'sst_mask', 'lat_ref', 'lon_ref']]
            # keep only the relevant columns
            df_sst = pd.concat([df_sst, df_l], ignore_index=True)
            break
        # load chl
        df_chl = pd.DataFrame()
        for f in tqdm(list_chl, desc='Loading CHL'):
            # load the data
            if not os.path.exists(f):
                continue
            df_l = pd.read_feather(f)
            # keep the data belonging to the dates in year_month_day
            df_l = df_l[df_l['date'].isin(year_month_day) & (df_l['chl1_mean_mask'] == False)][
                ['date', 'chl1_mean', 'chl1_mean_mask', 'lat_ref', 'lon_ref']]
            # keep only the relevant columns
            df_chl = pd.concat([df_chl, df_l], ignore_index=True)
            break

        # load sla
        df_sla = pd.DataFrame()
        for f in tqdm(list_sla, desc='Loading SLA'):
            # load the data
            if not os.path.exists(f):
                continue
            df_l = pd.read_feather(f)
            df_l = df_l[df_l['date'].isin(year_month_day) & (df_l['sla_mask'] == False)]
            # go through lat_ref and lon_ref and remove duplicates while keeping the one that corresponds to the one
            # with the lowest dist value
            df_l = df_l.sort_values(by=['lat_ref', 'lon_ref', 'dist'])
            df_l = df_l.drop_duplicates(subset=['lat_ref', 'lon_ref'], keep='first')
            # keep the data belonging to the dates in year_month_day
            df_l = df_l[df_l['date'].isin(year_month_day) & (df_l['sla_mask'] == False)][
                ['date', 'sla', 'sla_mask', 'eke', 'lat_ref', 'lon_ref']]
            # keep only the relevant columns
            df_sla = pd.concat([df_sla, df_l], ignore_index=True)

        print(df_sst.shape)
        # load chl


if __name__ == '__main__':
    dl = DataLoader([2010, 2018])
    dl.load_data()
