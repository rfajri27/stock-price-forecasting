"""
Transform data into a standardized format.

This module provides asynchronous transformation functions for:
- stock data
- company profile
- fundamentals data
- technicals data

It expects `helper` to define `logger`.
"""

from modules.helper import *
import numpy as np
import pandas as pd
import asyncio

logger = logging.getLogger(__name__)

async def transform_stock_data(stock_df:pd.DataFrame):
    """Transform stock data to a standardized format.

    Args:
        stock_data (pd.DataFrame): DataFrame containing historical OHLCV data.

    Returns:
        pd.DataFrame: Transformed DataFrame with adjusted features.
    
    Raises:
        ValueError: If error occurs during transformation.
    """
    try:
        stock_df["Date"] = stock_df["Date"].dt.date
    except Exception as e:
        logger.error(f"Error transforming stock data: {e}")
        raise ValueError(f"Error transforming stock data: {e}")
    
    return stock_df


async def transform_company_profile(company_profile_df:pd.DataFrame):
    """Transform company profile data to a standardized format.

    Args:
        company_profile (pd.DataFrame): DataFrame containing company profile data.

    Returns:
        pd.DataFrame: Transformed DataFrame with standardized features.
        
    Raises:
        ValueError: If error occurs during transformation.
    """
    try:
        company_profile_df["updateAt"] = pd.to_datetime(company_profile_df["updateAt"])
    except Exception as e:
        logger.error(f"Error transforming company profile data: {e}")
        raise ValueError(f"Error transforming company profile data: {e}")
    
    return company_profile_df

async def transform_fundamentals_data(fundamentals_df:pd.DataFrame):
    """Transform fundamentals data to a standardized format.

    Args:
        fundamentals_data (pd.DataFrame): DataFrame containing fundamentals data.

    Returns:
        pd.DataFrame: Transformed DataFrame with standardized features.
        
    Raises:
        ValueError: If error occurs during transformation.
    """
    try:
        fundamentals_df["updateAt"] = pd.to_datetime(fundamentals_df["updateAt"])
    except Exception as e:
        logger.error(f"Error transforming fundamentals data: {e}")
        raise ValueError(f"Error transforming fundamentals data: {e}")
    
    return fundamentals_df

async def transform_technicals_data(technicals_df:pd.DataFrame):
    """Transform technicals data to a standardized format.

    Args:
        technicals_data (pd.DataFrame): DataFrame containing technicals data.

    Returns:
        pd.DataFrame: Transformed DataFrame with standardized features.
        
    Raises:
        ValueError: If error occurs during transformation.
    """
    try:
        technicals_df["updateAt"] = pd.to_datetime(technicals_df["updateAt"])
    except Exception as e:
        logger.error(f"Error transforming technicals data: {e}")
        raise ValueError(f"Error transforming technicals data: {e}")
    
    return technicals_df

async def transform(stock_df:pd.DataFrame, company_profile_df:pd.DataFrame, fundamentals_df:pd.DataFrame, technicals_df:pd.DataFrame):
    """Transform all data into a unified format.

    Args:
        stock_df (pd.DataFrame): DataFrame containing historical OHLCV data.
        company_profile_df (pd.DataFrame): DataFrame containing company profile data.
        fundamentals_df (pd.DataFrame): DataFrame containing fundamentals data.
        technicals_df (pd.DataFrame): DataFrame containing technicals data.

    Returns:
        pd.DataFrame: Transformed DataFrame with adjusted features.
    """
    stock_df, company_profile_df, fundamentals_df, technicals_df = await asyncio.gather(
        transform_stock_data(stock_df),
        transform_company_profile(company_profile_df),
        transform_fundamentals_data(fundamentals_df),
        transform_technicals_data(technicals_df)
    )
    
    return stock_df, company_profile_df, fundamentals_df, technicals_df


if __name__ == "__main__":
    from extract import extract
    
    stock_df, company_profile_df, fundamentals_df, technicals_df = asyncio.run(extract(start_date="2025-08-01", end_date="2025-08-09"))
    stock_df, company_profile_df, fundamentals_df, technicals_df = asyncio.run(transform(stock_df, company_profile_df, fundamentals_df, technicals_df))
    
    print(stock_df.head())
    print(company_profile_df.head())
    print(fundamentals_df.head())
    print(technicals_df.head())