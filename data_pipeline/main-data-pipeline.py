"""
Main data pipeline to extract, transform, and load data into the database.
"""

import asyncio
import pandas as pd

from modules.extract import extract
from modules.transform import transform
from modules.load import load
from modules.helper import *

async def get_latest_date():
    """Get the latest date from the stock_data table"""
    
    query = f"""SELECT MAX("Date") AS "Date" FROM stock_data"""
    engine = get_db_engine()
    df = pd.read_sql_query(query, engine)
    return df.iloc[0]["Date"]

async def main():
    """Main function to run the data pipeline"""
    
    latest_date = await get_latest_date()
    if latest_date is None:
        start_date = "2006-01-01"
        end_date = "2025-08-09"
        
    if latest_date == current_date:
        logger.info("No new data to extract")
        return
    
    start_date = latest_date
    end_date = current_date
    
    logger.info(f"Extracting data from {start_date} to {end_date}")
    stock_df, company_profile_df, fundamentals_df, technicals_df = await extract(
        start_date=start_date, end_date=end_date)
    
    logger.info("Transforming data")
    stock_df, company_profile_df, fundamentals_df, technicals_df = await transform(
        stock_df, company_profile_df, fundamentals_df, technicals_df)
    
    stock_df = stock_df[stock_df["Date"] > latest_date]
    
    if stock_df.empty:
        logger.info("No new data to extract")
        return
    
    logger.info("Loading data")
    await load(stock_df, company_profile_df, fundamentals_df, technicals_df)

if __name__ == "__main__":
    asyncio.run(main())