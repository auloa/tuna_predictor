# The code below is for training the GAM model
# The data is loaded from the data_loader.py file

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pygam import LinearGAM, s, f, te
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from data_loader import DataLoader


def train_gam(year_range, plot_=False):
    dl = DataLoader(year_range)
    df = dl.load_data()
    X = df[['year', 'month', 'lat_ref', 'lon_ref', 'chl1_mean', 'sst', 'sla', 'eke', 'z', 'total_effort']]
    # offset_ = df[['total_effort']]
    y = df['total_catch']

    # split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    # standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # train the model
    gam = LinearGAM()

    gam.fit(X_train, y_train)

    predictions = gam.predict(X_test)
    print(predictions)
    print('R2 score: ', r2_score(y_test, predictions))
    print('RMSE: ', np.sqrt(mean_squared_error(y_test, predictions)))


if __name__ == '__main__':
    train_gam([2017])
