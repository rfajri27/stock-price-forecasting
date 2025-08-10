import pandas as pd
from prefect import task, get_run_logger
from helper import *


@task(name="data_ingestion")
async def data_ingestion() -> pd.DataFrame:
    """
    Ingest historical stock data from the database.

    Returns
    - pandas DataFrame with columns including Date, Open, High, Low, Close, Volume
    """
    logger = get_run_logger()
    engine = get_db_engine()
    
    query = f"""
    SELECT * FROM stock_data
    WHERE "Date" <= '{current_date}'
    """
    try:
        logger.info("Ingesting data from database")
        df = pd.read_sql_query(query, engine)
        logger.info(f"Ingested {len(df)} rows of data")
    except Exception as e:
        logger.error(f"Error ingesting data: {e}")
        raise
    
    return df

if __name__ == "__main__":
    import asyncio
    df = asyncio.run(data_ingestion())
    print(df.head())