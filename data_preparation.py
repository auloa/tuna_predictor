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

import netCDF4 as nc
import xarray as xr
import numpy as np
import pandas
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial import KDTree

from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors, KDTree
from utils.config_loader import Configs
from utils.distance_metric import haversine_metric
import regex as re

class DataReader:
    def __init__(self, year_range=[2017], ref_dataset='chl', use_haversine=True, testing=False,
                 ):

        self.df_lat_lon_pairs_ref = None
        self.lat_min_ref = None
        self.lat_max_ref = None
        self.lon_min_ref = None
        self.lon_max_ref = None
        self.df = None
        self.df_bath = None
        self.df_sst = None
        self.df_chl = None
        self.df_catch = None
        self.df_sla = None
        self.ref_dataset = ref_dataset
        self.use_haversine = use_haversine
        self.year_range = year_range
        if testing:
            self.year_range = range(self.year_range[0], self.year_range[0] + 1) if len(
                self.year_range) == 2 else self.year_range
            self.month_range = range(1, 2)
            self.day_range = range(1, 3)
        else:
            self.year_range = range(self.year_range[0], self.year_range[1] + 1) if len(
                self.year_range) == 2 else self.year_range
            self.month_range = range(1, 13)
            self.day_range = range(1, 32)

        if not isinstance(year_range, list):
            raise TypeError('year_range must be a list')

        if len(year_range) == 1:
            self.year_name = str(year_range[0])
        elif len(year_range) == 2:
            self.year_name = str(year_range[0]) + '_' + str(year_range[1])

        self.data_dir = Configs.get('DATA_DIR')
        self.chl_dir = os.path.join(self.data_dir, 'Globocolor')
        self.sst_dir = os.path.join(self.data_dir, 'SST')
        self.catch_dir = os.path.join(self.data_dir, 'catch')
        self.sla_dir = os.path.join(self.data_dir, 'SLA')
        self.bathymetry_dir = os.path.join(self.data_dir, 'bathymetry')
        self.knn_ref_ind = None

    def read_catch(self, reset=False):
        """
        Reading Catch and Effort Data.
        """
        path_catch = os.path.join(self.data_dir, f'catch_{self.year_name}.ftr')
        if os.path.exists(path_catch) and not reset:
            print('Reading catch and effort data from .ftr file...')
            self.df_catch = pd.read_feather(path_catch)
        else:
            print('Reading catch and effort data from csv files...')
            self.df_catch = pd.DataFrame()
            for year in tqdm(self.year_range, desc='Reading Data data: Year', position=0, leave=False):
                # list mission folders
                mission_folders = os.listdir(os.path.join(self.catch_dir, str(year)))
                # remove files from the list
                mission_folders = [x for x in mission_folders if not x.endswith('.ini')]
                if year > 2018:
                    mission_folders = ['']
                # loop through the mission folders
                for mission_folder in tqdm(mission_folders, desc='Reading Data data: Mission', position=0, leave=False):
                    # list the files in the mission folder
                    mission_files = os.listdir(os.path.join(self.catch_dir, str(year), mission_folder))
                    # loop through the mission files
                    for mission_file in mission_files:
                        # read the mission file
                        df_l = self.read_catch_file(year, mission_folder, mission_file)
                        # concatenate the data frames
                        self.df_catch = pd.concat([self.df_catch, df_l], ignore_index=True)
                # save the data frame to csv file
                print('\nsaving the data frame to .ftr file...')
                self.df_catch.reset_index(inplace=True)
                self.df_catch.drop(columns=['index'], inplace=True)
                self.df_catch.to_feather(path_catch)

        # set lat and lon range
        buffer = 0.5
        self.lat_min_ref = self.df_catch['lat'].min() - buffer
        self.lat_max_ref = self.df_catch['lat'].max() + buffer
        self.lon_min_ref = self.df_catch['lon'].min() - buffer
        self.lon_max_ref = self.df_catch['lon'].max() + buffer

    def read_catch_file(self, year, mission_folder, mission_file):
        # read the mission file
        if year > 2018:
            # use delimiter = ';' for the 2019  and onwards data
            df = pd.read_csv(os.path.join(self.catch_dir, str(year), mission_folder, mission_file), delimiter=';')
        else:
            # use the default delimiter for the 2018 and earlier data
            df = pd.read_csv(os.path.join(self.catch_dir, str(year), mission_folder, mission_file))
        # translate the column names to English and rename the columns list of column names: Lance
        # Barco	Viaje	TÃ©cnica	Fecha	Latitud	Longitud	AAA	BBCO	Bigeye	Bonita	Azul
        # Otras	Total	AltimetrÃ­a	Plancton	Temperatura	SubTemp	Termoclina	Corriente
        # Comentarios
        # drop columns with Nan or when all the values are same
        # df.dropna(axis=1, how='all', inplace=True)

        df.columns = ['lance', 'boat', 'trip', 'technique', 'date', 'lat', 'lon', 'aaa', 'bbco',
                      'bigeye', 'bonita', 'azul', 'otras', 'total_catch', 'altimetry', 'plankton',
                      'temperature', 'subTemp', 'thermocline', 'current', 'comments']

        # split the data column into data and time columns
        # for example date: 2017-01-25T16:43:00 -> date: 2017-01-25, time: 16:43:00
        df['date'] = df['date'].apply(lambda x: x.split('T'))
        df['time'] = df['date'].apply(lambda x: x[1])
        df['date'] = df['date'].apply(lambda x: x[0])

        # convert date format from year-month-day to day-month-year
        df = df.sort_values(by=['lat', 'lon'])
        df = self.extract_effort_data(df)
        return df

    def extract_effort_data(self, df, reset=False):
        # go through the df comments columns and extract the effort data
        # each datapoint may contain numbers in the string, extract the numbers and convert to int

        # if the effort data is already extracted, return
        if 'effort' in df.columns and not reset:
            return df

        # extract different fishing techniques
        # go through comments column and extract the unique methods
        comments = df['comments'].unique()
        key_words = [str(x) for x in comments]
        # join the comments
        key_words = ' '.join(key_words).replace(':', ' ').replace(',', ' ').replace('/', ' ')
        # split the comments
        key_words = set(key_words.split(' '))
        # remove any string that is a number of is nan
        key_words = [x for x in key_words if not x.isdigit() and x != 'nan' and x != '']

        #go through the key words and show sample comments
        for key_word in key_words:
            print(key_word)
            print(df[~df.comments.isnan()][df['comments'].str.contains(key_word)]['comments'].sample(5).values)



        df['effort'] = df['comments'].apply(self.process_comments)


    def process_comments(self, comment):
        # extract the effort data from the comments column
        effort = []
        # split the comment by ','
        comment = comment.split(',')
        # loop through the comment
        for c in comment:
            # extract the numbers from the comment
            c = re.findall(r'\d+', c)
            # loop through the numbers
            for n in c:
                # convert the number to int and append to the effort list
                effort.append(int(n))
        return effort

    def read_bathymetry(self, reset=False):
        path_bathymetry = os.path.join(self.data_dir, 'bathymetry.ftr')
        if os.path.exists(path_bathymetry) and not reset:
            return

        print('Loading bathymetry data...')
        file_name = 'ETOPO1_Bed_g_gmt4.grd'
        file_path = os.path.join(self.bathymetry_dir, file_name)

        dt = nc.Dataset(file_path)
        lat = dt.variables['y'][:]
        lon = dt.variables['x'][:]
        z = dt.variables['z'][:]

        # reshape lat and lon to 10801*21601 and then reshape to 1 dimensional data
        lat = lat.reshape(-1, 1).repeat(21601, axis=1).reshape(-1)
        lon = lon.reshape(1, -1).repeat(10801, axis=0).reshape(-1)

        # reshape z to 1 dimensional data
        z = z.reshape(-1)

        # join lat and lon to a single array and then truncate the data
        lat_lon_z = np.vstack((lat, lon, z)).T
        lat_lon_z = lat_lon_z[(lat_lon_z[:, 0] >= self.lat_min_ref) & (lat_lon_z[:, 0] <= self.lat_max_ref) &
                              (lat_lon_z[:, 1] >= self.lon_min_ref) & (lat_lon_z[:, 1] <= self.lon_max_ref)]

        # split the lat and lon
        lat = lat_lon_z[:, 0]
        lon = lat_lon_z[:, 1]
        z = lat_lon_z[:, 2]

        lat = lat[::20]
        lon = lon[::20]
        z = z[::20]

        # sample the data to reduce the size

        print('Creating the bathymetry dataframe...')
        # creat a data frame
        self.df_bath = pd.DataFrame({'lat': lat, 'lon': lon, 'z': z})

        print('\nsorting the dataframe and saving to csv file...')
        # sort self.d_bath by lat and lon
        print('\nsorting the data frame by lat, and lon')
        self.df_bath = self.df_bath.sort_values(by=['lat', 'lon'])

        print('\nsaving the data frame to .ftr file...')
        # save the data frame to .ftr file
        self.df_bath.reset_index(inplace=True)
        self.df_bath.drop(columns=['index'], inplace=True)

        self.df_bath.to_feather(path_bathymetry)

    def define_ref_lat_lon_pairs(self, reset=False):
        """
        Set the reference latitude and longitude from self.ref_dataset
        """
        df_ref = None
        # use exec function to use string in self.ref_dataset to choose the right df variable
        exec(f'df_ref = self.df_{self.ref_dataset}')
        if df_ref is None or len(df_ref) == 0:
            # load data from .ftr format
            # list the files in the data directory with self.ref_dataset in the name and pick the first one
            file_name = [file for file in glob.glob(os.path.join(self.data_dir, f'{self.ref_dataset}*.ftr'))][0]
            # read the data frame from .ftr file
            df_ref = pd.read_feather(file_name)
        path_lat_lon_pairs_ref = os.path.join(self.data_dir, f'lat_lon_pairs_{self.ref_dataset}.ftr')
        if os.path.exists(path_lat_lon_pairs_ref) and not reset:
            print('\nReading reference lat and lon pairs from feather file...')
            self.df_lat_lon_pairs_ref = pd.read_feather(path_lat_lon_pairs_ref)
        else:
            # extract unique lat and lon pairs from df_ref
            self.df_lat_lon_pairs_ref = df_ref[['lat', 'lon']].drop_duplicates()
            # sort the lat and lon pairs
            self.df_lat_lon_pairs_ref = self.df_lat_lon_pairs_ref.sort_values(by=['lat', 'lon'])
            # reset the index
            self.df_lat_lon_pairs_ref.reset_index(inplace=True)
            # drop the index column
            self.df_lat_lon_pairs_ref.drop(columns=['index'], inplace=True)
            # save the reference lat and lon pairs to ftr file
            self.df_lat_lon_pairs_ref.to_feather(path_lat_lon_pairs_ref)

    def read_chl(self, reset=False):
        """
        Read chlorophyll data from Globocolor
        """
        file_list = glob.glob(os.path.join(self.data_dir, 'chl*.ftr'))
        if len(file_list) > 0 and not reset:
            return

        sorted_ = False
        # read data from .nc files and get data frame and merge the data frames
        print('\nReading chlorophyll data from .nc files and compiling into pandas data frame format')
        self.df_chl = pd.DataFrame()
        for year in tqdm(self.year_range, desc='Reading chlorophyll data: Year', position=0, leave=False):
            for month in tqdm(self.month_range, f'year: {year}', position=0, leave=False):
                for day in tqdm(self.day_range, f'year: {year}, month: {month}'):
                    try:
                        df_l = self.read_chl_single_day(year, month, day)
                        df_l['date'] = pd.to_datetime(f'{year}-{month}-{day}').strftime('%Y-%m-%d')
                        if not sorted_:
                            # get the index of the sorted data frame
                            sorted_index = df_l.sort_values(by=['lat', 'lon']).index
                            sorted_ = True
                        # concatenate the df_l to df
                        self.df_chl = pd.concat([self.df_chl, df_l], ignore_index=True)
                    except Exception as e:
                        print(e)
                # sort the data frame by lat and
                print('sorting the data frame by lat, lon, and date')
                # print('sorting the data frame by lat, lon, and date')
                # use the cyclic property of the data to sort the data frame
                # get the length of the self.df_chl and the length of the sorted_index
                len_df = len(self.df_chl)
                len_sorted_index = len(sorted_index)
                # get the number of times the sorted_index should be repeated
                n = len_df // len_sorted_index
                # repeat the sorted_index n times while adding len_sorted_index to indices at each step
                sorted_index_ex = np.tile(sorted_index, n) + np.array(
                    [[i] * len_sorted_index for i in range(0, len_df, len_sorted_index)]).flatten()
                # sort the data frame by the sorted_index
                self.df_chl = self.df_chl.loc[sorted_index_ex]
                # save the data frame to csv file
                print('saving the chl data frame to csv file')
                self.df_chl.reset_index(inplace=True)
                self.df_chl.drop(columns=['index'], inplace=True)
                # save the data frame using feather format
                self.df_chl.to_feather(os.path.join(self.data_dir, f'chl_{year}_{month}.ftr'))
                self.df_chl = pd.DataFrame()

    def read_chl_single_day(self, year, month, day):
        file_name = f'L3m_{year}{month:02d}{day:02d}__GLOB_25_GSM-MODVIR_CHL1_DAY_00.nc'
        file_path = os.path.join(self.chl_dir, str(year), f'{month:02d}', f'{day:02d}', file_name)
        dt = nc.Dataset(file_path)
        lat = dt.variables['lat'][:]
        lon = dt.variables['lon'][:]

        # reshape lat and lon to 720*1440
        lat = lat.reshape(-1, 1).repeat(1440, axis=1)
        lon = lon.reshape(1, -1).repeat(720, axis=0)

        # reshape the lat and long to single dimension
        lat = lat.reshape(-1)
        lon = lon.reshape(-1)

        chl1_mean_data = dt.variables['CHL1_mean'][:]
        chl1_mean = chl1_mean_data.data.reshape(-1)
        chl1_mean_mask = chl1_mean_data.mask.reshape(-1)
        chl1_flags = dt.variables['CHL1_flags'][:].data.reshape(-1)
        chl1_error = dt.variables['CHL1_error'][:]
        chl1_error_vaue = chl1_error.data.reshape(-1)
        chl1_error_mask = chl1_error.mask.reshape(-1)
        df = pd.DataFrame({'lat': lat, 'lon': lon,
                           'chl1_mean': chl1_mean, 'chl1_mean_mask': chl1_mean_mask,
                           # 'chl1_flags': chl1_flags, 'chl1_error': chl1_error_vaue,
                           # 'chl1_error_mask': chl1_error_mask
                           })

        # filter the data frame to only include the data within the lat and lon range
        df = self.trim_df(df)
        return df

    def read_sst(self, reset=False):
        """
        Read SST data from Globocolor
        """
        file_list = glob.glob(os.path.join(self.data_dir, 'sst*.ftr'))
        if len(file_list) > 0 and not reset:
            return
        # read data from .nc files and get data frame and merge the data frames
        print('\nReading sst data from .nc files and compiling it into csv format')
        self.df_sst = pd.DataFrame()
        for year in tqdm(self.year_range, desc='Reading SST data: Year', position=0, leave=False):
            for month in tqdm(self.month_range, f'year: {year}', position=0, leave=False):
                for day in tqdm(self.day_range, f'year: {year}, month: {month}'):
                    try:
                        df_l = self.read_sst_single_day(year, month, day)
                        df_l['date'] = pd.to_datetime(f'{year}-{month}-{day}').strftime('%Y-%m-%d')
                        # concatenate the df_l to df
                        self.df_sst = pd.concat([self.df_sst, df_l], ignore_index=True)
                    except Exception as e:
                        print(e)
                if len(self.df_sst) == 0:
                    continue
                # save the data frame to csv file
                print('save the sst data frame to .ftr file')
                self.df_sst.reset_index(inplace=True)
                self.df_sst.drop(columns=['index'], inplace=True)
                # save the data frame using feather format
                self.df_sst.to_feather(os.path.join(self.data_dir, f'sst_{year}_{month}.ftr'))
                self.df_sst = pd.DataFrame()

    def read_sst_single_day(self, year, month, day):
        file_name = f'{year}{month:02d}{day:02d}120000-NCEI-L4_GHRSST-SSTblend-AVHRR_OI-GLOB-v02.0-fv02.1.nc'
        file_path = os.path.join(self.sst_dir, str(year), f'{month:02d}', f'{day:02d}', file_name)
        dt = nc.Dataset(file_path)
        lat = dt.variables['lat'][:]
        lon = dt.variables['lon'][:]

        # reshape lat and lon to 720*1440
        lat = lat.reshape(-1, 1).repeat(1440, axis=1)
        lon = lon.reshape(1, -1).repeat(720, axis=0)

        # reshape the lat and long to single dimension
        lat = lat.reshape(-1)
        lon = lon.reshape(-1)

        data_time = dt.variables['time'][:].data[0]
        time_unit = dt.variables['time'].units
        # convert the time to datetime
        data_time = nc.num2date(data_time, time_unit, calendar='standard').isoformat()
        # only keep the time and not date
        data_time = data_time.split('T')[1]

        sst_data = dt.variables['analysed_sst'][:]
        sst = sst_data.data.reshape(-1)
        sst_mask = sst_data.mask.reshape(-1)

        sst_analysis_error = dt.variables['analysis_error'][:].data.reshape(-1)

        sea_ice_fraction = dt.variables['sea_ice_fraction'][:].data.reshape(-1)
        sea_ice_fraction_mask = dt.variables['sea_ice_fraction'][:].mask.reshape(-1)

        df = pd.DataFrame({'lat': lat, 'lon': lon,
                           'sst': sst, 'sst_mask': sst_mask,
                           # 'sst_analysis_error': sst_analysis_error,
                           # 'sea_ice_fraction': sea_ice_fraction,
                           # 'sea_ice_fraction_mask': sea_ice_fraction_mask
                           })
        df['time'] = data_time

        df = self.trim_df(df)

        return df

    def read_sla(self, reset=False):
        """
        Read SLA data from AVISO
        """
        file_list = glob.glob(os.path.join(self.data_dir, 'sla*.ftr'))
        if len(file_list) > 0 and not reset:
            return

        print('\nReading sla data from .nc files and compiling into csv format')
        # read data from .nc files and get data frame and merge the data frames
        self.df_sla = pd.DataFrame()
        for year in tqdm(self.year_range, desc='Reading SLA data: Year', position=0, leave=False):
            for month in tqdm(self.month_range, f'year: {year}', position=0, leave=False):
                for day in tqdm(self.day_range, f'year: {year}, month: {month}'):
                    try:
                        df_l = self.read_sla_single_day(year, month, day)
                        df_l['date'] = pd.to_datetime(f'{year}-{month}-{day}').strftime('%Y-%m-%d')
                        # concatenate the df_l to df
                        self.df_sla = pd.concat([self.df_sla, df_l], ignore_index=True)
                    except Exception as e:
                        print(e)
                print('save the sla data frame to .ftr file')
                self.df_sla.reset_index(inplace=True)
                self.df_sla.drop(columns=['index'], inplace=True)
                # save the data frame using feather format
                self.df_sla.to_feather(os.path.join(self.data_dir, f'sla_{year}_{month}.ftr'))
                self.df_sla = pd.DataFrame()

    def read_sla_single_day(self, year, month, day):
        file_name = f'dt_global_allsat_phy_l4_{year}{month:02d}{day:02d}_20210318.nc'
        file_path = os.path.join(self.sla_dir, file_name)
        dt = nc.Dataset(file_path)
        lat = dt.variables['latitude'][:].data
        lon = dt.variables['longitude'][:].data
        # lat_bnds = dt.variables['lat_bnds'][:].data
        # lon_bnds = dt.variables['lon_bnds'][:].data

        # reshape lat and lon to 1700*3600 and then reshape to single dimension
        lat = lat.reshape(-1, 1).repeat(3600, axis=1).reshape(-1)
        lon = lon.reshape(1, -1).repeat(1700, axis=0).reshape(-1)

        # lat_bnds = lat_bnds.reshape(-1, 2).repeat(3600, axis=1).reshape(-1, 2)
        # lon_bnds = lon_bnds.reshape(1, -1).repeat(1700, axis=0).reshape(-1, 2)

        # lat_bnds_l = lat_bnds[:, 0]
        # lat_bnds_u = lat_bnds[:, 1]
        #
        # lon_bnds_l = lon_bnds[:, 0]
        # lon_bnds_u = lon_bnds[:, 1]
        #
        # adt = dt.variables['adt'][:].data.reshape(-1)  # adt is the absolute dynamic topography
        # adt_mask = dt.variables['adt'][:].mask.reshape(-1)
        sla = dt.variables['sla'][:].data.reshape(-1)
        sla_mask = dt.variables['sla'][:].mask.reshape(-1)
        # ugos = dt.variables['ugos'][:].data.reshape(-1)
        # ugos_mask = dt.variables['ugos'][:].mask.reshape(-1)
        # vgos = dt.variables['vgos'][:].data.reshape(-1)
        # vgos_mask = dt.variables['vgos'][:].mask.reshape(-1)
        ugosa = dt.variables['ugosa'][:].data.reshape(-1)
        ugosa_mask = dt.variables['ugosa'][:].mask.reshape(-1)
        vgosa = dt.variables['vgosa'][:].data.reshape(-1)
        # vgosa_mask = dt.variables['vgosa'][:].mask.reshape(-1)

        # calculate eke from ugosa and vgosa
        eke = np.sqrt(ugosa ** 2 + vgosa ** 2)
        eke_mask = ugosa_mask

        df = pd.DataFrame({'lat': lat, 'lon': lon,
                           # 'lat_bnds_l': lat_bnds_l,
                           # 'lon_bnds_u': lon_bnds_u,
                           # 'lon_bnds_l': lon_bnds_l,
                           # 'lat_bnds_u': lat_bnds_u,
                           # 'adt': adt, 'adt_mask': adt_mask,
                           'sla': sla, 'sla_mask': sla_mask,
                           # 'ugos': ugos, 'ugos_mask': ugos_mask,
                           # 'vgos': vgos, 'vgos_mask': vgos_mask,
                           # 'ugosa': ugosa, 'ugosa_mask': ugosa_mask,
                           # 'vgosa': vgosa, 'vgosa_mask': vgosa_mask,
                           'eke': eke, 'eke_mask': eke_mask})

        df = self.trim_df(df)

        # # reduce the size of the data frame by resampling the data frame by skipping one row without averaging
        # df = df.iloc[::2, :]
        return df

    def trim_df(self, df):
        # filter the data frame to only include the data within the lat and lon range
        df = df[(df['lat'] >= self.lat_min_ref) & (df['lat'] <= self.lat_max_ref) &
                (df['lon'] >= self.lon_min_ref) & (df['lon'] <= self.lon_max_ref)]

        # reset the index of the data frame
        df.reset_index(inplace=True)
        df.drop(columns=['index'], inplace=True)
        return df

    def assign_lat_lon_ref(self,
                           dataset_name):  # assume that the data is cyclic, so that the reference lat and lon pairs are the same as the data lat and lon pairs
        # check the length of dr.df_chl and df_lat_lon_pairs_chl
        # if the length is the same or the length is multiple of the length of df_lat_lon_pairs_chl, then the data is cyclic
        # otherwise, the data is not cyclic
        # get the list of all the data files in data_dir with the dataset_name in the file name
        print('\nAssigning lat and lon reference for ' + dataset_name + '...')
        data_files = glob.glob(os.path.join(self.data_dir, dataset_name + '*.ftr'))
        # read the data files
        df_lat_lon_pairs = pd.DataFrame(columns=['lat', 'lon'])
        for data_file in data_files:
            df = pd.read_feather(data_file)
            df_lat_lon = df.groupby(['lat', 'lon']).size().reset_index()[['lat', 'lon']]
            # compare df_lat_lon with df_lat_lon_pairs_ref and add any new lat and lon pairs to df_lat_lon_pairs
            df_lat_lon_pairs = pd.concat([df_lat_lon_pairs, df_lat_lon]).drop_duplicates().reset_index(drop=True)

        df_lat_lon_pairs = self.mapping_lat_lon(df_lat_lon_pairs)

        # save the lat and lon pairs to .ftr file
        df_lat_lon_pairs.to_feather(os.path.join(self.data_dir, f'lat_lon_pairs_{dataset_name}.ftr'))

    def mapping_lat_lon(self, df_lat_lon_pairs_query):
        if self.use_haversine:
            if self.knn_ref_ind is None:
                self.knn_ref_ind = NearestNeighbors(n_neighbors=1, algorithm='auto', metric=haversine_metric).fit(
                    self.df_lat_lon_pairs_ref[['lat', 'lon']].values)
        else:
            # use kd tree to find the nearest neighbor
            self.kd_tree_ref = KDTree(self.df_lat_lon_pairs_ref[['lat', 'lon']].values)

        print('Mapping the lat and lon from query data to reference data...')

        # check if lat and lon are equal in both data frames
        lat_check = df_lat_lon_pairs_query['lat'].values == self.df_lat_lon_pairs_ref['lat'].values
        if isinstance(lat_check, bool):
            lat_check = np.array([lat_check])
        lon_check = df_lat_lon_pairs_query['lon'].values == self.df_lat_lon_pairs_ref['lon'].values
        if isinstance(lon_check, bool):
            lon_check = np.array([lon_check])

        if lat_check.all() and lon_check.all():
            print('lat and lon are equal in both data frames')
            df_lat_lon_pairs_query['lat_ref'] = df_lat_lon_pairs_query['lat'].values
            df_lat_lon_pairs_query['lon_ref'] = df_lat_lon_pairs_query['lon'].values
            df_lat_lon_pairs_query['dist'] = 0
        else:
            # look for pattern
            def knn_query(row):
                if self.use_haversine:
                    dist, ind = self.knn_ref_ind.kneighbors([[row['lat'], row['lon']]])
                else:
                    dist, ind = self.kd_tree_ref.query([[row['lat'], row['lon']]], k=1)
                return dist[0], ind[0]

            # apply knn_query function to each row of DataFrame to get distance and index of the closest coordinate
            tqdm.pandas()
            dist_ind = df_lat_lon_pairs_query.progress_apply(lambda row: pd.Series(knn_query(row)), axis=1).values

            # add lat_ref, lon_ref and dist columns to the data frame
            df_lat_lon_pairs_query['lat_ref'] = self.df_lat_lon_pairs_ref.iloc[dist_ind[:, 1]]['lat'].values
            df_lat_lon_pairs_query['lon_ref'] = self.df_lat_lon_pairs_ref.iloc[dist_ind[:, 1]]['lon'].values
            df_lat_lon_pairs_query['dist'] = [x[0] for x in dist_ind[:, 0]]

        return df_lat_lon_pairs_query

    def set_ref_lat_lon(self, dataset_name):
        print(f'Setting the reference lat and lon to the {dataset_name} data...')

        lat_lon_pair_file = os.path.join(self.data_dir, f'lat_lon_pairs_{dataset_name}.ftr')
        if os.path.exists(lat_lon_pair_file):
            df_lat_lon_pairs = pd.read_feather(lat_lon_pair_file)
        else:
            raise Exception(f'lat_lon_pairs_{dataset_name}.ftr does not exist in {self.data_dir}')

        data_files = glob.glob(os.path.join(self.data_dir, dataset_name + '*.ftr'))
        # read the data files
        for data_file in tqdm(data_files, desc='Assigning lat_ref and lon_ref to ' + dataset_name + ' data...'):
            df = pd.read_feather(data_file)
            # add index column
            df = df.reset_index()

            # define function to return index of the df where the lat and lon match the lat and lon from lat_lon_pairs_query
            def lat_lon_match(row):
                ind_q = df_lat_lon_pairs[
                    (df_lat_lon_pairs['lat'] == row['lat']) & (df_lat_lon_pairs['lon'] == row['lon'])].index.tolist()
                return ind_q[0]

            tqdm.pandas()
            ind_q_list = df.progress_apply(lambda row: pd.Series(lat_lon_match(row)), axis=1).values

            ind_q_list = [x[0] for x in ind_q_list]

            # set the reference lat and lon to the query data
            df['lat_ref'] = df_lat_lon_pairs.loc[ind_q_list, 'lat_ref'].values.tolist()
            df['lon_ref'] = df_lat_lon_pairs.loc[ind_q_list, 'lon_ref'].values.tolist()
            df['dist'] = df_lat_lon_pairs.loc[ind_q_list, 'dist'].values.tolist()

            # drop the index column
            df = df.drop(columns=['index'])

            df.to_feather(data_file)

    def set_ref_lat_lon_cyclic(self, dataset_name):
        # assign lat_ref and lon_ref to dr.df_chl
        # assume that the data is cyclic, so that the reference lat and lon pairs are the same as the data lat and lon pairs
        # check the length of dr.df_chl and df_lat_lon_pairs_chl
        # if the length is the same or the length is multiple of the length of df_lat_lon_pairs_chl, then the data is cyclic
        # otherwise, the data is not cyclic
        # get the list of all the data files in data_dir with the dataset_name in the file name
        lat_lon_pair_file = os.path.join(self.data_dir, f'lat_lon_pairs_{dataset_name}.ftr')
        if os.path.exists(lat_lon_pair_file):
            df_lat_lon_pairs = pd.read_feather(lat_lon_pair_file)
        else:
            raise Exception(f'lat_lon_pairs_{dataset_name}.ftr does not exist in {self.data_dir}')

        data_files = glob.glob(os.path.join(self.data_dir, dataset_name + '*.ftr'))
        # read the data files
        for data_file in tqdm(data_files, desc='Assigning lat_ref and lon_ref to ' + dataset_name + ' data...'):
            df = pd.read_feather(data_file)
            print(len(df), len(df_lat_lon_pairs), len(df) / len(df_lat_lon_pairs))
            if dataset_name == 'chl' or dataset_name == 'sst':
                df['lat_ref'] = df['lat']
                df['lon_ref'] = df['lon']
                df['dist'] = 0
                df.to_feather(data_file)
                continue

            # check if the length of df is the same as the length of df_lat_lon_pairs
            if len(df) == len(df_lat_lon_pairs):
                df['lat_ref'] = df_lat_lon_pairs['lat_ref']
                df['lon_ref'] = df_lat_lon_pairs['lon_ref']
                df['dist'] = df_lat_lon_pairs['dist']
            elif len(df) % len(df_lat_lon_pairs) == 0:
                # get the number of cycles
                num_cycles = len(df) // len(df_lat_lon_pairs)
                df_lat_lon_pairs_ = pd.concat([df_lat_lon_pairs] * num_cycles, ignore_index=True)
                df['lat_ref'] = df_lat_lon_pairs_['lat_ref']
                df['lon_ref'] = df_lat_lon_pairs_['lon_ref']
                df['dist'] = df_lat_lon_pairs_['dist']
            else:
                # the data is not cyclic, raise error
                raise ValueError('The data is not cyclic')

            # save the data file
            df.to_feather(data_file)

    def prepare_final_data(self):
        df = pd.read_feather(os.path.join(self.data_dir, f'catch_{self.year_name}.ftr'))
        # add the bathymetry data to the df
        df_bath = pd.read_feather(os.path.join(self.data_dir, 'bathymetry.ftr'))
        # take the z values from the bathymetry data and add it to the df by matching lat_ref and lon_ref columns
        tqdm.pandas()
        df['z'] = df.progress_apply(lambda row: df_bath[(df_bath['lat_ref'] == row['lat_ref']) &
                                                        (df_bath['lon_ref'] == row['lon_ref'])]['z'].values[0], axis=1)

        # define dataset list
        dataset_list = ['chl', 'sst', 'sla']

        # extract the unique dates in the dataset
        dates = df['date'].unique()
        year_month = []
        year_month_day = []
        # get the list of unique year month combinations in the dates
        for date in dates:
            # get the year and month in string format
            date = datetime.strptime(date, "%Y-%m-%d")
            year = str(date.year)
            month = str(date.month)
            day = str(date.day).zfill(2)
            year_month.append(year + '_' + month)
            year_month_day.append(year + '-' + month.zfill(2) + '-' + day)
        year_month = list(set(year_month))

        column_dict = {'sla': ['sla', 'sla_mask', 'eke', 'eke_mask', 'date', 'lat_ref', 'lon_ref'],
                       'chl': ['chl1_mean', 'chl1_mean_mask', 'date', 'lat_ref', 'lon_ref'],
                       'sst': ['sst', 'sst_mask', 'date', 'lat_ref', 'lon_ref']}

        # go through the year_month combinations and open the data files
        # add the data from the file to the final data file by matching the data and the lat_ref and lon_ref format
        for year_month_ in tqdm(year_month, desc='Preparing final data...'):
            for dataset_name in dataset_list:
                # get the data file
                data_file = os.path.join(self.data_dir, f'{dataset_name}_{year_month_}.ftr')
                if os.path.exists(data_file):
                    df_data = pd.read_feather(data_file)
                    # add the data to the final data file by matching lat_ref, lon_ref and date
                    df = df.merge(df_data[column_dict[dataset_name]],
                                  on=['lat_ref', 'lon_ref', 'date'],
                                  how='left',
                                  suffixes=('', '_y'))
                    # drop the columns with _y suffix and merge the values with the original column by taking the
                    # non-null values
                    for column in column_dict[dataset_name]:
                        if column + '_y' in df.columns:
                            df[column] = df.apply(lambda row: row[column] if pd.notnull(row[column]) else row[column + '_y'], axis=1)
                            df.drop(columns=[column + '_y'], inplace=True)
                else:
                    print(f'{data_file} does not exist for {dataset_name} and {year_month_}')

        # split the date column into year, month and day columns
        df['year'] = df['date'].apply(lambda x: x.split('-')[0])
        df['month'] = df['date'].apply(lambda x: x.split('-')[1])
        df['day'] = df['date'].apply(lambda x: x.split('-')[2])

        # save the final data file
        df.to_feather(os.path.join(self.data_dir, f'catch_{self.year_name}_final.ftr'))


def plot_catch_data(df_catch):
    import pandas as pd
    from shapely.geometry import Point
    import geopandas as gpd
    from geopandas import GeoDataFrame

    geometry = [Point(xy) for xy in zip(df_catch['lon'], df_catch['lat'])]
    gdf = GeoDataFrame(df_catch, geometry=geometry)
    geometry_ref = [Point(xy) for xy in zip(df_catch['lon_ref'], df_catch['lat_ref'])]
    gdf_ref = GeoDataFrame(df_catch, geometry=geometry_ref)

    # this is a simple map that goes with geopandas
    figure, ax = plt.subplots(figsize=(10, 6))
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world.plot(ax=ax, figsize=(10, 6))
    # plort gdf and gdf_ref on the same map
    gdf.plot(ax=ax, marker='o', color='red', markersize=3)
    gdf_ref.plot(ax=ax, marker='o', color='blue', markersize=3)
    plt.show()


if __name__ == '__main__':
    # testing = False
    dr = DataReader([2017], ref_dataset='chl', use_haversine=False)
    # # process the catch data on the first run
    # # first choose the reference lat and lon for the whole dataset
    # # for the prototyping purpose, lat and lon ranges used in the chl dataset are used
    # # first read the catch data and get the ranges for the lat and lon
    # # read the catch data
    dr.read_catch(reset=True)
    # dr.read_chl()
    # dr.read_sst()
    # dr.read_sla()
    # dr.read_bathymetry()
    # # extract the unique lat and lon pairs from the catch data and save them to a csv file as reference
    dr.define_ref_lat_lon_pairs(reset=False)
    datasets = ['catch']  # , 'chl', 'sst', 'sla', 'bath']
    for dataset in datasets:
        dr.assign_lat_lon_ref(dataset)
        if dataset == 'catch':
            dr.set_ref_lat_lon(dataset)
        else:
            dr.set_ref_lat_lon_cyclic(dataset)
    # exit()

    # Once all the data is processed, prepare the final data for training load the catch data, go through each data
    # point, find the corresponding data in the other datasets using lat_ref and lat_lon
    # Check the date associated with the datapoint in the catch data, open the corresponding data file, match the
    # lat_ref and lon_ref, and extract the data
    # save the data to a new file
    dr.prepare_final_data()
