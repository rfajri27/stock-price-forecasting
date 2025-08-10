from modules.helper import *
import pandas as pd


logger = logging.getLogger(__name__)

async def scaling(scaler, df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale the data.

    Parameters
    - df: pandas DataFrame with columns including Date, Close

    Returns
    - df: pandas DataFrame with columns including Date, Close
    """
    logger.info("Scaling data")
    df = df.copy()
    df["Close"] = scaler.transform(df["Close"].values.reshape(-1, 1))
    return df

def data_preprocessing_regression(df: pd.DataFrame) -> tuple:
    """
    Preprocess the data for regression model.

    Parameters
    - df: pandas DataFrame with columns including Date, Close

    Returns
    - input_x: pandas DataFrame with columns including Date, Close, lag_1, lag_2, lag_3, lag_4, lag_5, rolling_mean_3
    - output_y: pandas Series with columns including target
    """
    logger.info("Preprocessing data for regression")
    
    for lag in range(1, 6):  # Last 5 days
        df[f"lag_{lag}"] = df["Close"].shift(lag)
    
    df["rolling_mean_3"] = df["Close"].shift(1).rolling(window=3).mean()
    df["target"] = df["Close"].shift(-1)
    
    df = df[["Date", "Close", "lag_1", "lag_2", 
             "lag_3", "lag_4", "lag_5", "rolling_mean_3", "target"]]
    
    input_x = df[["Date", "Close", "lag_1", "lag_2", 
                  "lag_3", "lag_4", "lag_5", "rolling_mean_3"]]
    output_y = df["target"]
    return input_x, output_y