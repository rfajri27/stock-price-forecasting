"""
Load data into the database.

This module provides asynchronous loading functions for:
- stock data
- company profile
- fundamentals data
- technicals data

It expects `helper` to define `logger` and `get_db_engine`.
"""

from modules.helper import *
import pandas as pd
import asyncio

async def load_stock_data(engine, stock_df:pd.DataFrame):
    """Load stock data into the database.
    
    Args:
        stock_df (pd.DataFrame): DataFrame containing historical OHLCV data.
    """
    try:
        stock_df.to_sql("stock_data", engine, if_exists="append", index=False)
    except Exception as e:
        logger.error(f"Error loading stock data into the database: {e}")
        raise ValueError(f"Error loading stock data into the database: {e}")

async def load_company_profile_data(engine, company_profile_df:pd.DataFrame):
    """Load company profile data into the database.
    
    Args:
        company_profile_df (pd.DataFrame): DataFrame containing company profile data.
    """
    try:
        company_profile_df.to_sql("company_profile", engine, if_exists="append", index=False)
    except Exception as e:
        logger.error(f"Error loading company profile data into the database: {e}")
        raise ValueError(f"Error loading company profile data into the database: {e}")

async def load_fundamentals_data(engine, fundamentals_df:pd.DataFrame):
    """Load fundamentals data into the database.
    
    Args:
        fundamentals_df (pd.DataFrame): DataFrame containing fundamentals data.
    """
    try:
        fundamentals_df.to_sql("fundamentals", engine, if_exists="append", index=False)
    except Exception as e:
        logger.error(f"Error loading fundamentals data into the database: {e}")
        raise ValueError(f"Error loading fundamentals data into the database: {e}")

async def load_technicals_data(engine, technicals_df:pd.DataFrame):
    """Load technicals data into the database.
    
    Args:
        technicals_df (pd.DataFrame): DataFrame containing technicals data.
    """
    try:
        technicals_df.to_sql("technicals", engine, if_exists="append", index=False)
    except Exception as e:
        logger.error(f"Error loading technicals data into the database: {e}")
        raise ValueError(f"Error loading technicals data into the database: {e}")

async def load(stock_df:pd.DataFrame, company_profile_df:pd.DataFrame, fundamentals_df:pd.DataFrame, technicals_df:pd.DataFrame):
    """Load data into the database.

    Args:
        stock_df (pd.DataFrame): DataFrame containing historical OHLCV data.
        company_profile_df (pd.DataFrame): DataFrame containing company profile data.
        fundamentals_df (pd.DataFrame): DataFrame containing fundamentals data.
        technicals_df (pd.DataFrame): DataFrame containing technicals data.
    """
    try:
        engine = get_db_engine()
        await asyncio.gather(
            load_stock_data(engine, stock_df),
            load_company_profile_data(engine, company_profile_df),
            load_fundamentals_data(engine, fundamentals_df),
            load_technicals_data(engine, technicals_df)
        )
    except Exception as e:
        logger.error(f"Error loading data into the database: {e}")
        raise ValueError(f"Error loading data into the database: {e}")

if __name__ == "__main__":
    from extract import extract
    from transform import transform
    import asyncio
    
    stock_df, company_profile_df, fundamentals_df, technicals_df = asyncio.run(extract(start_date="2025-08-01", end_date="2025-08-09"))
    stock_df, company_profile_df, fundamentals_df, technicals_df = asyncio.run(transform(stock_df, company_profile_df, fundamentals_df, technicals_df))
    asyncio.run(load(stock_df, company_profile_df, fundamentals_df, technicals_df))