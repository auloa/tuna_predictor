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


class DataReader:
    def __init__(self, year_range, testing=False,
                 ):

        self.df = None
        self.df_bath = None
        self.df_sst = None
        self.df_chl = None
        self.df_catch_data = None
        self.df_sla = None
        self.year_range = year_range
        if testing:
            self.year_range = self.year_range[:1]
            self.month_range = range(1, 2)
            self.day_range = range(1, 3)
        else:
            self.month_range = range(1, 13)
            self.day_range = range(1, 32)
        if not isinstance(year_range, list):
            raise TypeError('year_range must be a list')
        if len(year_range) == 1:
            year_name = str(year_range[0])
        elif len(year_range) == 2:
            year_name = str(year_range[0]) + '_' + str(year_range[1])

        self.data_dir = Configs.get('DATA_DIR')
        self.chl_dir = os.path.join(self.data_dir, 'Globocolor')
        self.chl_csv = os.path.join(self.data_dir, f'chl_{year_name}.csv')
        self.sst_dir = os.path.join(self.data_dir, 'SST')
        self.sst_csv = os.path.join(self.data_dir, f'sst_{year_name}.csv')
        self.catch_dir = os.path.join(self.data_dir, 'catch')
        self.catch_csv = os.path.join(self.data_dir, f'catch_{year_name}.csv')
        self.sla_dir = os.path.join(self.data_dir, 'SLA')
        self.sla_csv = os.path.join(self.data_dir, f'sla_{year_name}.csv')
        self.bathymetry_dir = os.path.join(self.data_dir, 'bathymetry')
        self.bathymetry_csv = os.path.join(self.data_dir, 'bathymetry.csv')

    def read_chl(self):
        """
        Read chlorophyll data from Globocolor
        """
        if os.path.exists(self.chl_csv):
            print('\nReading chlorophyll data from csv file')
            self.df_chl = pd.read_csv(self.chl_csv)
            return None
        # read data from .nc files and get data frame and merge the data frames
        print('\nReading chlorophyll data from .nc files and compiling into csv format')

        self.df_chl = pd.DataFrame()

        for year in tqdm(self.year_range, desc='Reading chlorophyll data: Year', position=0, leave=False):
            for month in tqdm(self.month_range, f'year: {year}', position=0, leave=False):
                for day in tqdm(self.day_range, f'year: {year}, month: {month}'):
                    try:
                        df_l = self.read_chl_single_day(year, month, day)
                        df_l['date'] = pd.to_datetime(f'{year}-{month}-{day}')
                        # concatenate the df_l to df
                        self.df_chl = pd.concat([self.df_chl, df_l], ignore_index=True)
                    except Exception as e:
                        print(e)

        # sort the data frame by lat and lon
        # print('sorting the data frame by lat, lon, and date')
        self.df_chl.sort_values(by=['lat', 'lon'], inplace=True)
        # extract uniques dates in the data
        dates = self.df_chl.date.unique()
        df = pd.DataFrame()
        for date_ in dates:
            # extract df data part that belongs to date_
            df = pd.concat([df, self.df_chl[self.df_chl['date'] == date_]], ignore_index=True)

        self.df_chl = df
        # save the data frame to csv file
        self.df_chl.to_csv(self.chl_csv, index=False)

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
        df = pd.DataFrame({'lat': lat, 'lon': lon, 'chl1_mean': chl1_mean, 'chl1_mean_mask': chl1_mean_mask,
                           'chl1_flags': chl1_flags, 'chl1_error': chl1_error_vaue,
                           'chl1_error_mask': chl1_error_mask})

        return df

    def read_sst(self):
        """
        Read SST data from Globocolor
        """
        if os.path.exists(self.sst_csv):
            print('\nLoading sst data from csv')
            self.df_sst = pd.read_csv(self.sst_csv)
        else:
            # read data from .nc files and get data frame and merge the data frames
            print('\nReading sst data from .nc files and compiling it into csv format')
            self.df_sst = pd.DataFrame()
            for year in tqdm(self.year_range, desc='Reading SST data: Year', position=0, leave=False):
                for month in tqdm(self.month_range, f'year: {year}', position=0, leave=False):
                    for day in tqdm(self.day_range, f'year: {year}, month: {month}'):
                        try:
                            df_l = self.read_sst_single_day(year, month, day)
                            df_l['date'] = pd.to_datetime(f'{year}-{month}-{day}')
                            # concatenate the df_l to df
                            self.df_sst = pd.concat([self.df_sst, df_l], ignore_index=True)
                        except Exception as e:
                            print(e)

            # sort the data frame by lat and lon
            print('\nsorting the data frame by lat, lon, and date')
            self.df_sst.sort_values(by=['lat', 'lon'], inplace=True)
            # extract uniques dates in the data
            dates = self.df_sst.date.unique()
            df = pd.DataFrame()
            for date_ in dates:
                # extract df data part that belongs to date_
                df = pd.concat([df, self.df_sst[self.df_sst['date'] == date_]], ignore_index=True)

            self.df_sst = df
            # save the data frame to csv file
            self.df_sst.to_csv(self.sst_csv, index=False)

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

        df = pd.DataFrame({'lat': lat, 'lon': lon, 'sst': sst, 'sst_mask': sst_mask,
                           'sst_analysis_error': sst_analysis_error,
                           'sea_ice_fraction': sea_ice_fraction,
                           'sea_ice_fraction_mask': sea_ice_fraction_mask})
        df['time'] = data_time

        return df

    def read_sla(self):
        """
        Read SLA data from AVISO
        """
        if os.path.exists(self.sla_csv):
            print('\nLoading sla data from csv file')
            self.df_sla = pd.read_csv(self.sla_csv)
        else:
            print('\nReading sla data from .nc files and compiling into csv format')
            # read data from .nc files and get data frame and merge the data frames
            self.df_sla = pd.DataFrame()
            for year in tqdm(self.year_range, desc='Reading SLA data: Year', position=0, leave=False):
                for month in tqdm(self.month_range, f'year: {year}', position=0, leave=False):
                    for day in tqdm(self.day_range, f'year: {year}, month: {month}'):
                        try:
                            df_l = self.read_sla_single_day(year, month, day)
                            df_l['date'] = pd.to_datetime(f'{year}-{month}-{day}')
                            # concatenate the df_l to df
                            self.df_sla = pd.concat([self.df_sla, df_l], ignore_index=True)
                        except Exception as e:
                            print(e)

            # # sort the data frame by lat and lon
            # print('\nsorting the data frame by lat, lon, and date')
            # self.df_sla.sort_values(by=['lat', 'lon'], inplace=True)
            # # extract uniques dates in the data
            # dates = self.df_sla.date.unique()
            # df = pd.DataFrame()
            # for date_ in dates:
            #     # extract df data part that belongs to date_
            #     df = pd.concat([df, self.df_sla[self.df_sla['date'] == date_]], ignore_index=True)
            # self.df_sla = df

            # save the data frame to csv file
            self.df_sla.to_csv(self.sla_csv, index=False)

    def read_sla_single_day(self, year, month, day):
        file_name = f'dt_global_allsat_phy_l4_{year}{month:02d}{day:02d}_20210318.nc'
        file_path = os.path.join(self.sla_dir, file_name)
        dt = nc.Dataset(file_path)
        lat = dt.variables['latitude'][:].data
        lon = dt.variables['longitude'][:].data
        lat_bnds = dt.variables['lat_bnds'][:].data
        lon_bnds = dt.variables['lon_bnds'][:].data

        # reshape lat and lon to 1700*3600 and then reshape to single dimension
        lat = lat.reshape(-1, 1).repeat(3600, axis=1).reshape(-1)
        lon = lon.reshape(1, -1).repeat(1700, axis=0).reshape(-1)

        lat_bnds = lat_bnds.reshape(-1, 2).repeat(3600, axis=1).reshape(-1, 2)
        lon_bnds = lon_bnds.reshape(1, -1).repeat(1700, axis=0).reshape(-1, 2)

        lat_bnds_l = lat_bnds[:, 0]
        lat_bnds_u = lat_bnds[:, 1]

        lon_bnds_l = lon_bnds[:, 0]
        lon_bnds_u = lon_bnds[:, 1]

        adt = dt.variables['adt'][:].data.reshape(-1)  # adt is the absolute dynamic topography
        adt_mask = dt.variables['adt'][:].mask.reshape(-1)
        sla = dt.variables['sla'][:].data.reshape(-1)
        sla_mask = dt.variables['sla'][:].mask.reshape(-1)
        ugos = dt.variables['ugos'][:].data.reshape(-1)
        ugos_mask = dt.variables['ugos'][:].mask.reshape(-1)
        vgos = dt.variables['vgos'][:].data.reshape(-1)
        vgos_mask = dt.variables['vgos'][:].mask.reshape(-1)
        ugosa = dt.variables['ugosa'][:].data.reshape(-1)
        ugosa_mask = dt.variables['ugosa'][:].mask.reshape(-1)
        vgosa = dt.variables['vgosa'][:].data.reshape(-1)
        vgosa_mask = dt.variables['vgosa'][:].mask.reshape(-1)

        # calculate eke from ugosa and vgosa
        eke = np.sqrt(ugosa ** 2 + vgosa ** 2)
        eke_mask = ugosa_mask

        df = pd.DataFrame({'lat': lat, 'lon': lon,
                           'lat_bnds_l': lat_bnds_l,
                           'lon_bnds_u': lon_bnds_u,
                           'lon_bnds_l': lon_bnds_l,
                           'lat_bnds_u': lat_bnds_u,
                           'adt': adt, 'adt_mask': adt_mask,
                           'sla': sla, 'sla_mask': sla_mask,
                           'ugos': ugos, 'ugos_mask': ugos_mask,
                           'vgos': vgos, 'vgos_mask': vgos_mask,
                           'ugosa': ugosa, 'ugosa_mask': ugosa_mask,
                           'vgosa': vgosa, 'vgosa_mask': vgosa_mask,
                           'eke': eke, 'eke_mask': eke_mask})

        return df

    def read_bathymetry(self):
        print('Loading bathymetry data...')
        file_name = 'ETOPO1_Bed_g_gmt4.grd'
        file_path = os.path.join(self.bathymetry_dir, file_name)

        if os.path.exists(self.bathymetry_csv):
            self.df_bath = pd.read_csv(self.bathymetry_csv)
            # self.df_bath = self.df_bath[:1000]
            # self.df_bath.to_csv(os.path.join(self.data_dir, 'bathymetry_testing.csv'), index=False)
        else:

            dt = nc.Dataset(file_path)
            lat = dt.variables['y'][:]
            lon = dt.variables['x'][:]
            z = dt.variables['z'][:]

            # reshape lat and lon to 10801*21601 and then reshape to 1 dimensional data
            lat = lat.reshape(-1, 1).repeat(21601, axis=1).reshape(-1)
            lon = lon.reshape(1, -1).repeat(10801, axis=0).reshape(-1)

            # reshape z to 1 dimensional data
            z = z.reshape(-1)

            buffer = 0.01
            # get the starting lat and lon from self.df_sst and truncate the data
            lat_min = self.df_sst['lat'].min() - buffer
            lat_max = self.df_sst['lat'].max() + buffer
            lon_min = self.df_sst['lon'].min() - buffer
            lon_max = self.df_sst['lon'].max() + buffer

            # join lat and lon to a single array and then truncate the data
            lat_lon_z = np.vstack((lat, lon, z)).T
            lat_lon_z = lat_lon_z[(lat_lon_z[:, 0] >= lat_min) & (lat_lon_z[:, 0] <= lat_max) &
                                  (lat_lon_z[:, 1] >= lon_min) & (lat_lon_z[:, 1] <= lon_max)]

            # split the lat and lon
            lat = lat_lon_z[:, 0]
            lon = lat_lon_z[:, 1]
            z = lat_lon_z[:, 2]

            lat = lat[::225]
            lon = lon[::225]
            z = z[::225]

            # sample the data to reduce the size
            print(len(lat))

            print('Creating the bathymetry dataframe...')
            # creat a data frame
            self.df_bath = pd.DataFrame({'lat': lat, 'lon': lon, 'z': z})

            print('\nsorting the dataframe and saving to csv file...')
            # sort self.d_bath by lat and lon
            print('\nsorting the data frame by lat, and lon')
            self.df_bath = self.df_bath.sort_values(by=['lat', 'lon'])

            self.df_bath.to_csv(self.bathymetry_csv, index=False)

    def read_catch_data(self):
        """

        Reading Catch and Effort Data.

        """

        if os.path.exists(self.catch_csv):
            print('Loading catch and effort data...')
            # read the catch and effort data
            self.df_catch_data = pd.read_csv(self.catch_csv)
        else:
            print('Reading catch and effort data from csv files...')
            self.df_catch_data = pd.DataFrame()
            for year in tqdm(self.year_range, desc='Reading Data data: Year', position=0, leave=False):
                # list mission folders
                mission_folders = os.listdir(os.path.join(self.catch_dir, str(year)))
                # remove files from the list
                mission_folders = [x for x in mission_folders if not x.endswith('.ini')]
                # loop through the mission folders
                for mission_folder in tqdm(mission_folders, desc='Reading Data data: Mission', position=0, leave=False):
                    # list the files in the mission folder
                    mission_files = os.listdir(os.path.join(self.catch_dir, str(year), mission_folder))
                    # loop through the mission files
                    for mission_file in mission_files:
                        # read the mission file
                        df_l = self.read_catch_data_file(year, mission_folder, mission_file)
                        # concatenate the data frames
                        self.df_catch_data = pd.concat([self.df_catch_data, df_l], ignore_index=True)

            # sort the data frame by lat and lon and then by date
            print('\nsorting the data frame by lat, lon and date')
            self.df_catch_data = self.df_catch_data.sort_values(by=['lat', 'lon'])
            # extract uniques dates in the data
            dates = self.df_catch_data.date.unique()
            df = pd.DataFrame()
            for date_ in dates:
                # extract df data part that belongs to date_
                df = pd.concat([df, self.df_catch_data[self.df_catch_data['date'] == date_]], ignore_index=True)

            self.df_catch_data = df

            # save the data frame to csv file
            self.df_catch_data.to_csv(self.catch_csv, index=False)

    def read_catch_data_file(self, year, mission_folder, mission_file):
        # read the mission file
        df = pd.read_csv(os.path.join(self.catch_dir, str(year), mission_folder, mission_file))
        # translate the column names to English and rename the columns list of column names: Lance
        # Barco	Viaje	TÃ©cnica	Fecha	Latitud	Longitud	AAA	BBCO	Bigeye	Bonita	Azul
        # Otras	Total	AltimetrÃ­a	Plancton	Temperatura	SubTemp	Termoclina	Corriente
        # Comentarios

        df.columns = ['lance', 'boat', 'trip', 'technique', 'date', 'lat', 'lon', 'aaa', 'bbco',
                      'bigeye', 'bonita', 'azul', 'otras', 'total_catch', 'altimetry', 'plankton',
                      'temperature', 'subTemp', 'thermocline', 'current', 'comments']

        # split the data column into data and time columns
        # for example date: 2017-01-25T16:43:00 -> date: 25-01-17, time: 16:43:00
        df['date'] = df['date'].apply(lambda x: x.split('T'))
        df['time'] = df['date'].apply(lambda x: x[1])
        df['date'] = df['date'].apply(lambda x: x[0])

        # convert date format from year-month-day to day-month-year
        df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').strftime('%d-%m-%y'))

        return df


def haversine(lat1, lon1, lat2, lon2):
    R = 6372.8
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    a = sin(dLat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dLon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return R * c


def haversine_metric(x, y):
    return haversine(x[0], x[1], y[0], y[1])


def mapping_lat_lon(lat_lon_pairs_ref, lat_lon_pairs_query, knn):
    print('Mapping the lat and lon from query data to reference data...')

    def knn_query(row):
        dist, ind = knn.kneighbors([[row['lat'], row['lon']]])
        return dist[0], ind[0]

    df = pd.DataFrame(lat_lon_pairs_query, columns=['lat', 'lon'])
    # apply knn_query function to each row of DataFrame to get distance and index of the closest coordinate
    tqdm.pandas()
    dist_ind = df.progress_apply(lambda row: pd.Series(knn_query(row)), axis=1).values

    return dist_ind


if __name__ == '__main__':
    testing = True
    dr = DataReader([2017], testing=testing)

    path_df_chl_sst_bath = os.path.join(dr.data_dir, 'df_final.csv')
    if not os.path.exists(path_df_chl_sst_bath):
        # reference lat and lon pairs are taken from dr.df_chl and dr.df_sst
        dr.read_chl()
        dr.read_sst()
        dr.read_bathymetry()
        dr.read_catch_data()
        dr.read_sla()

        # if testing:
        #     dr.df_chl = dr.df_chl[:1000]
        #     dr.df_sst = dr.df_sst[:1000]
        #     dr.df_sla = dr.df_sla[:1000]
        #     dr.df_bath = dr.df_bath[:1000]
        #     dr.df_catch_data = dr.df_catch_data[:1000]

        print('Mapping the lat and lon from bathymetry data to chl and sst data...')
        # get the unique lat and lon pairs from the data frames
        lat_lon_pairs_chl = dr.df_chl.groupby(['lat', 'lon']).size().reset_index()[['lat', 'lon']].values
        lat_lon_pairs_bath = dr.df_bath.groupby(['lat', 'lon']).size().reset_index()[['lat', 'lon']].values

        dist_ind_chl_bath_path = os.path.join(dr.data_dir, 'dist_ind_chl_bath.npy')

        knn_ref_ind = None

        if os.path.exists(dist_ind_chl_bath_path):
            dist_ind = np.load(dist_ind_chl_bath_path)
        else:
            print('Creating the knn model...')
            # create knn model with haversine distance function
            knn_ref_ind = NearestNeighbors(n_neighbors=1, algorithm='auto', metric=haversine_metric).fit(
                lat_lon_pairs_chl)
            dist_ind = mapping_lat_lon(lat_lon_pairs_chl, lat_lon_pairs_bath, knn_ref_ind)
            # save the dist_ind array for future use
            np.save(os.path.join(dr.data_dir, 'dist_ind_chl_bath.npy'), dist_ind)

        # add lat_ref, lon_ref columns to dr.df_bath
        # go through lat_lon_pairs_bath and set corresponding lat and lon from dr.df_chl
        dr.df_bath['lat_ref'] = np.nan
        dr.df_bath['lon_ref'] = np.nan
        dr.df_bath['dist'] = np.nan
        for i in tqdm(range(len(lat_lon_pairs_bath)), "setting reference lat and lon (bathymetry)"):
            dr.df_bath.loc[
                (dr.df_bath['lat'] == lat_lon_pairs_bath[i][0]) & (dr.df_bath['lon'] == lat_lon_pairs_bath[i][1]), [
                    'lat_ref', 'lon_ref', 'dist']] = np.concatenate(
                [lat_lon_pairs_chl[dist_ind[i, 1]][0], dist_ind[i, 0]])

        dr.df_bath.to_csv(dr.bathymetry_csv, index=False)

        # map the closest lat and lon from dr.df_chl and dr.df_sla
        print('Mapping the lat and lon from sla data to chl and sst data...')
        lat_lon_pairs_sla = dr.df_sla[['lat', 'lon']].values

        dist_ind_chl_sla_path = os.path.join(dr.data_dir, 'dist_ind_chl_sla.npy')
        if os.path.exists(dist_ind_chl_sla_path):
            dist_ind = np.load(dist_ind_chl_sla_path)
        else:
            if knn_ref_ind is None:
                knn_ref_ind = NearestNeighbors(n_neighbors=1, algorithm='auto', metric=haversine_metric).fit(
                    lat_lon_pairs_chl)
            dist_ind = mapping_lat_lon(lat_lon_pairs_chl, lat_lon_pairs_sla, knn_ref_ind)
            # save the dist_ind array for future use
            np.save(os.path.join(dr.data_dir, 'dist_ind_chl_sla.npy'), dist_ind)

        # add lat_ref, lon_ref columns to dr.df_sla
        # go through lat_lon_pairs_sla and set corresponding lat and lon from dr.df_chl
        dr.df_sla['lat_ref'] = np.nan
        dr.df_sla['lon_ref'] = np.nan
        dr.df_sla['dist'] = np.nan
        for i in tqdm(range(len(lat_lon_pairs_sla)), "setting reference lat and lon (sla)"):
            dr.df_sla.loc[
                (dr.df_sla['lat'] == lat_lon_pairs_sla[i][0]) & (dr.df_sla['lon'] == lat_lon_pairs_sla[i][1]), [
                    'lat_ref', 'lon_ref', 'dist']] = np.concatenate(
                [lat_lon_pairs_chl[dist_ind[i, 1]][0], dist_ind[i, 0]])

        dr.df_sla.to_csv(dr.sla_csv, index=False)

        # map the closest lat, lon in lat_lon_pairs_bath to dr.df_catch_data
        print('Mapping the lat and lon from catch data to chl data...')
        lat_lon_pairs_catch = dr.df_catch_data[['lat', 'lon']].values

        dist_ind_chl_catch_path = os.path.join(dr.data_dir, 'dist_ind_chl_catch.npy')
        if os.path.exists(dist_ind_chl_catch_path):
            dist_ind = np.load(dist_ind_chl_catch_path)
        else:
            if knn_ref_ind is None:
                knn_ref_ind = NearestNeighbors(n_neighbors=1, algorithm='auto', metric=haversine_metric).fit(
                    lat_lon_pairs_chl)
            dist_ind = mapping_lat_lon(lat_lon_pairs_chl, lat_lon_pairs_catch, knn_ref_ind)

        # add lat_ref, lon_ref columns to dr.df_catch_data
        # go through lat_lon_pairs_catch and set corresponding lat and lon from dr.df_chl
        dr.df_catch_data['lat_ref'] = np.nan
        dr.df_catch_data['lon_ref'] = np.nan
        dr.df_catch_data['dist'] = np.nan

        for i in range(len(lat_lon_pairs_catch)):
            dr.df_catch_data.loc[
                (dr.df_catch_data['lat'] == lat_lon_pairs_catch[i][0]) & (
                        dr.df_catch_data['lon'] == lat_lon_pairs_catch[i][1]), [
                    'lat_ref', 'lon_ref', 'dist']] = np.concatenate(
                [lat_lon_pairs_chl[dist_ind[i, 1]][0], dist_ind[i, 0]])

        dr.df_catch_data.to_csv(dr.catches_csv, index=False)

        # Add lat_ref and lon_ref columns to dr.df_chl and dr.df_sst
        dr.df_chl['lat_ref'] = dr.df_chl['lat']
        dr.df_chl['lon_ref'] = dr.df_chl['lon']
        dr.df_chl['dist'] = 0
        dr.df_sst['lat_ref'] = dr.df_sst['lat']
        dr.df_sst['lon_ref'] = dr.df_sst['lon']
        dr.df_sst['dist'] = 0

        # save the data frames
        dr.df_chl.to_csv(dr.chl_csv, index=False)
        dr.df_sst.to_csv(dr.sst_csv, index=False)

        # Extract usable data out of all the dataframes
        # list the variables that will be used for modeling:
        # lat, lon, date, sst, chl_mean, sla, eke, z, Total


    else:
        df = pd.read_csv(path_df_chl_sst_bath)
        # map the closest lat and lon from dr.df_chl and dr.df_sla
        print('Mapping the lat and lon from sla data to chl and sst data...')

        lat_lon_pairs_sla = dr.df_sla[['lat', 'lon']].values
        lat_lon_pairs_chl_sst = df[['lat', 'lon']].values

        dist_ind = mapping_lat_lon(lat_lon_pairs_chl_sst, lat_lon_pairs_sla)

        dr.read_bathymetry()
        # map the closest sla lat, lon to the lat and lon in df
        dr.read_catch_data()
