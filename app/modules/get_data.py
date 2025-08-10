from modules.helper import *
import pandas as pd

logger = logging.getLogger(__name__)

def get_stock_data() -> pd.DataFrame:
    """
    Get the stock data from the database.

    Returns
    - df: pandas DataFrame with columns including Date, Close
    """
    logger.info("Getting stock data")
    engine = get_db_engine()
    query = f"""
    select "Date", "Close" from stock_data
    where "Date" <= '{current_date}'
    """
    try:
        df = pd.read_sql_query(query, engine)
        logger.info(f"Got {len(df)} rows of stock data")
    except Exception as e:
        logger.error(f"Error getting stock data: {e}")
        raise
    
    return df

def get_new_stock_data() -> pd.DataFrame:
    """
    Get the stock data from the database.

    Returns
    - df: pandas DataFrame with columns including Date, Close
    """
    logger.info("Getting stock data")
    engine = get_db_engine()
    query = f"""
    select "Date", "Close" from stock_data
    where "Date" > (select max("Date") from stock_data_predicted)
    """
    try:
        df = pd.read_sql_query(query, engine)
        logger.info(f"Got {len(df)} rows of stock data")
    except Exception as e:
        logger.error(f"Error getting stock data: {e}")
        raise
    
    return df

def get_predicted_stock_data(from_date: str = None) -> pd.DataFrame:
    """
    Get the predicted stock data from the database.

    Returns
    - df: pandas DataFrame with columns including Date, Predicted_Close
    """
    logger.info("Getting predicted stock data")
    engine = get_db_engine()
    query = f"""
    select "Date", "Predicted_Close" from stock_data_predicted
    where "Date" >= '{from_date}'
    order by "Date"
    """
    try:
        df = pd.read_sql_query(query, engine)
        logger.info(f"Got {len(df)} rows of predicted stock data")
    except Exception as e:
        logger.error(f"Error getting predicted stock data: {e}")
        raise
    
    return df

def get_stock_data_with_volume(from_date: str = None) -> pd.DataFrame:
    """
    Get the stock data with volume from the database.

    Returns
    - df: pandas DataFrame with columns including Date, Close, Volume
    """
    logger.info("Getting stock data with volume")
    engine = get_db_engine()
    query = f"""
    select "Date", "Close", "Volume" from stock_data
    where "Date" >= '{from_date}'
    order by "Date"
    """
    try:
        df = pd.read_sql_query(query, engine)
        logger.info(f"Got {len(df)} rows of stock data with volume")
    except Exception as e:
        logger.error(f"Error getting stock data with volume: {e}")
        raise
    
    return df