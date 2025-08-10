"""
This module contains the tasks for preprocessing the data.

The tasks are:
- training_test_split: Split the data into train and test sets
- data_preprocessing_regression: Preprocess the data for regression model
- data_preprocessing_lstm: Preprocess the data for LSTM model
- main_data_preprocessing: Main function to preprocess the data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import asyncio
import joblib
from prefect import task, get_run_logger
from helper import *

@task(name="train_test_split")
async def training_test_split(df: pd.DataFrame, prop_train=0.8) -> tuple:
    """
    Splits a time series into train and test sets based on a proportion.
    
    Parameters:
        df (pd.DataFrame): The time series data.
        prop_train (float): Proportion of data to use for training (default 0.8).
    
    Returns:
        tuple: (train_series, test_series)
    
    Raises:
        ValueError: If the input sequence is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input sequence must be a pandas DataFrame.")
    
    split_point = round(prop_train * len(df))
    train, test = df[:split_point+1], df[split_point+1:]
    
    return train, test.reset_index(drop=True)

@task(name="data_preprocessing_regression")
async def data_preprocessing_regression(df: pd.DataFrame) -> tuple:
    """
    Preprocess the data for regression model.

    Parameters
    - df: pandas DataFrame with columns including Date, Open, High, Low, Close, Volume

    Returns
    - input_x: pandas DataFrame with columns including Close, lag_1, lag_2, lag_3, lag_4, lag_5, rolling_mean_3
    - output_y: pandas Series with columns including target
    """
    logger = get_run_logger()
    logger.info("Preprocessing data for regression")
    
    for lag in range(1, 6):  # Last 5 days
        df[f"lag_{lag}"] = df["Close"].shift(lag)
    
    df["rolling_mean_3"] = df["Close"].shift(1).rolling(window=3).mean()
    df["target"] = df["Close"].shift(-1)
    
    df = df[["Close", "lag_1", "lag_2", 
             "lag_3", "lag_4", "lag_5", "rolling_mean_3", "target"]].dropna()
    
    input_x = df[["Close", "lag_1", "lag_2", 
                  "lag_3", "lag_4", "lag_5", "rolling_mean_3"]]
    output_y = df["target"]
    return input_x, output_y

async def split_sequence(seq, n_steps, flatten_target=True) -> tuple:
    """
    Transforms a univariate time series into input/output sequences for time series prediction.
    
    Parameters:
        seq (pd.Series or np.ndarray): Time series values.
        n_steps (int): Number of time steps to use as input.
        flatten_target (bool): Whether to flatten the output array (default True).
        
    Returns:
        tuple: (X, y) where X is shape (samples, n_steps), and y is (samples,) or (samples, 1)
    """
    X, y = [], []

    if isinstance(seq, pd.Series):
        seq = seq.values  # convert to np.array

    for i in range(len(seq) - n_steps):
        X.append(seq[i:i + n_steps])
        y.append(seq[i + n_steps])

    X = np.array(X)
    y = np.array(y)

    if flatten_target:
        y = y.ravel()
        
    return X, y

@task(name="data_preprocessing_lstm")
async def data_preprocessing_lstm(df: pd.DataFrame) -> tuple:
    """
    Preprocess the data for LSTM model.

    Parameters
    - df: pandas DataFrame with columns including Date, Open, High, Low, Close, Volume

    Returns
    - input_x: numpy array with shape (samples, n_steps)
    - output_y: numpy array with shape (samples,)
    """
    logger = get_run_logger()
    logger.info("Preprocessing data for LSTM")
    
    input_x, output_y = await split_sequence(
        seq=df["Close"],
        n_steps=5
    )
    return input_x, output_y

@task(name="main_data_preprocessing")
async def main_data_preprocessing(df: pd.DataFrame) -> tuple:
    """
    Main function to preprocess the data.

    Parameters
    - df: pandas DataFrame with columns including Date, Open, High, Low, Close, Volume

    Returns
    - training_set_regression: tuple of (input_x, output_y)
    - training_set_lstm: tuple of (input_x, output_y)
    - test_set_regression: tuple of (input_x, output_y)
    - test_set_lstm: tuple of (input_x, output_y)
    """
    logger = get_run_logger()
    logger.info("Preprocessing data")
    
    training_df, test_df = await training_test_split(df)
    
    scaler = RobustScaler()
    scaler.fit(training_df["Close"].values.reshape(-1, 1))
    joblib.dump(scaler, "artifacts/scaler.joblib")
    training_df["Close"] = scaler.transform(training_df["Close"].values.reshape(-1, 1))
    test_df["Close"] = scaler.transform(test_df["Close"].values.reshape(-1, 1))
    
    training_set_regression, test_set_regression = await asyncio.gather(
        data_preprocessing_regression(training_df),
        data_preprocessing_regression(test_df)
    )
    
    training_set_lstm, test_set_lstm = await asyncio.gather(
        data_preprocessing_lstm(training_df),
        data_preprocessing_lstm(test_df)
    )
    
    return training_set_regression, training_set_lstm, test_set_regression, test_set_lstm


if __name__ == "__main__":
    from data_ingestion import data_ingestion
    df = asyncio.run(data_ingestion())
    training_set_regression, training_set_lstm, test_set_regression, test_set_lstm = asyncio.run(main_data_preprocessing(df))
    print(training_set_regression[0].head())
    print("training_set_regression: ", training_set_regression[0].shape, training_set_regression[1].shape)
    print("test_set_regression: ", test_set_regression[0].shape, test_set_regression[1].shape)
    print("training_set_lstm: ", training_set_lstm[0].shape, training_set_lstm[1].shape)
    print("test_set_lstm: ", test_set_lstm[0].shape, test_set_lstm[1].shape)