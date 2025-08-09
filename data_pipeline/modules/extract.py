"""
Asynchronous extraction utilities for fetching stock data via yfinance.

This module provides coroutine functions to retrieve:
- historical OHLCV price data
- company profile metadata
- fundamentals metrics
- technical/market statistics

It expects `helper` to define `STOCK_NAME`, `logger`, and `current_date_time`.
"""
from modules.helper import *
import numpy as np
import pandas as pd
import json
import yfinance as yf
import asyncio

logger = logging.getLogger(__name__)

async def get_stock_data(ticker, start_date:str, end_date:str):
    """Fetch historical OHLCV data for a ticker within a date range.

    Args:
        ticker (yfinance.Ticker): Ticker object for the target stock.
        start_date (str): Start date in ISO format YYYY-MM-DD.
        end_date (str): End date in ISO format YYYY-MM-DD (exclusive in yfinance).

    Returns:
        pandas.DataFrame: DataFrame containing the historical OHLCV data with a reset index.

    Raises:
        ValueError: If no data is returned by the API.
    """
    try:
        stock_data = ticker.history(start=start_date, end=end_date).reset_index()
        logger.info(f"Successfully fetched stock data for {ticker.info['longName']} between {start_date} and {end_date}.")
        
    except Exception as e:
        logger.error(f"Error fetching stock data: {e}")
        raise ValueError(f"No data returned for '{ticker.info['longName']}' between {start_date} and {end_date}.")
    
    return stock_data

async def get_company_profile(ticker):
    """Fetch company profile information for the configured stock.

    Args:
        ticker (yfinance.Ticker): Ticker object for the target stock.

    Returns:
        pandas.DataFrame: Single-row DataFrame containing company profile fields.

    Raises:
        ValueError: If profile information cannot be retrieved.
    """
    try:
        company_profile = {
            "stockName": STOCK_NAME,
            "longName": ticker.info["longName"],
            "sector": ticker.info["sector"],
            "industry": ticker.info["industry"],
            "longBusinessSummary": ticker.info["longBusinessSummary"],
            "country": ticker.info["country"],
            "city": ticker.info["city"],
            "address": ticker.info["address1"],
            "address2": ticker.info["address2"],
            "zip": ticker.info["zip"],
            "phone": ticker.info["phone"],
            "website": ticker.info["website"],
            "fullTimeEmployees": ticker.info["fullTimeEmployees"],
            "companyOfficers": json.dumps(ticker.info["companyOfficers"]),
            "updateAt": current_date_time,
        }

        company_profile_df = pd.DataFrame([company_profile])
        
        logger.info(f"Successfully fetched company profile for {ticker.info['longName']}.")
        return company_profile_df
    except Exception as e:
        logger.error(f"Error fetching company profile: {e}")
        raise ValueError(f"No data returned for '{ticker.info['longName']}'.")

async def get_fundamentals_data(ticker):
    """Fetch fundamental metrics for the configured stock.

    Args:
        ticker (yfinance.Ticker): Ticker object for the target stock.

    Returns:
        pandas.DataFrame: Single-row DataFrame containing fundamental indicators and ratios.

    Raises:
        ValueError: If fundamentals cannot be retrieved.
    """
    try:
        fundamentals_data = {
            "stockName": STOCK_NAME,
            "marketCap": ticker.info["marketCap"],
            "enterpriseValue": ticker.info["enterpriseValue"],
            "enterpriseToRevenue": ticker.info["enterpriseToRevenue"],
            "trailingPE": ticker.info["trailingPE"],
            "forwardPE": ticker.info["forwardPE"],
            "priceToBook": ticker.info["priceToBook"],
            "priceToSalesTrailing12Months": ticker.info["priceToSalesTrailing12Months"],
            "trailingPegRatio": ticker.info["trailingPegRatio"],
            "netIncomeToCommon": ticker.info["netIncomeToCommon"],
            "trailingEps": ticker.info["trailingEps"],
            "forwardEps": ticker.info["forwardEps"],
            "epsCurrentYear": ticker.info["epsCurrentYear"],
            "priceEpsCurrentYear": ticker.info["priceEpsCurrentYear"],
            "earningsGrowth": ticker.info["earningsGrowth"],
            "revenueGrowth": ticker.info["revenueGrowth"],
            "earningsQuarterlyGrowth": ticker.info["earningsQuarterlyGrowth"],
            "profitMargins": ticker.info["profitMargins"],
            "returnOnAssets": ticker.info["returnOnAssets"],
            "returnOnEquity": ticker.info["returnOnEquity"],
            "operatingMargins": ticker.info["operatingMargins"],
            "grossMargins": ticker.info["grossMargins"],
            "ebitdaMargins": ticker.info["ebitdaMargins"],
            "totalRevenue": ticker.info["totalRevenue"],
            "revenuePerShare": ticker.info["revenuePerShare"],
            "grossProfits": ticker.info["grossProfits"],
            "totalCash": ticker.info["totalCash"],
            "totalCashPerShare": ticker.info["totalCashPerShare"],
            "operatingCashflow": ticker.info["operatingCashflow"],
            "totalDebt": ticker.info["totalDebt"],
            "dividendRate": ticker.info["dividendRate"],
            "dividendYield": ticker.info["dividendYield"],
            "fiveYearAvgDividendYield": ticker.info["fiveYearAvgDividendYield"],
            "payoutRatio": ticker.info["payoutRatio"],
            "lastDividendValue": ticker.info["lastDividendValue"],
            "lastDividendDate": ticker.info["lastDividendDate"],
            "trailingAnnualDividendRate": ticker.info["trailingAnnualDividendRate"],
            "trailingAnnualDividendYield": ticker.info["trailingAnnualDividendYield"],
            "exDividendDate": ticker.info["exDividendDate"],
            "sharesOutstanding": ticker.info["sharesOutstanding"],
            "floatShares": ticker.info["floatShares"],
            "heldPercentInsiders": ticker.info["heldPercentInsiders"],
            "heldPercentInstitutions": ticker.info["heldPercentInstitutions"],
            "updateAt": current_date_time,
        }

        fundamentals_df = pd.DataFrame([fundamentals_data])
        logger.info(f"Successfully fetched fundamentals data for {ticker.info['longName']}.")
        return fundamentals_df
    except Exception as e:
        logger.error(f"Error fetching fundamentals data: {e}")
        raise ValueError(f"No data returned for '{ticker.info['longName']}'.")
    
async def get_technicals_data(ticker):
    """Fetch technical and market statistics for the configured stock.

    Args:
        ticker (yfinance.Ticker): Ticker object for the target stock.

    Returns:
        pandas.DataFrame: Single-row DataFrame containing technical levels and market stats.

    Raises:
        ValueError: If technicals cannot be retrieved.
    """
    try:
        technicals_data = {
            "stockName": STOCK_NAME,
            "currentPrice": ticker.info["currentPrice"], 
            "previousClose": ticker.info["previousClose"], 
            "open": ticker.info["open"],
            "dayLow": ticker.info["dayLow"], 
            "dayHigh": ticker.info["dayHigh"], 
            "regularMarketDayLow": ticker.info["regularMarketDayLow"], 
            "regularMarketDayHigh": ticker.info["regularMarketDayHigh"],
            "fiftyTwoWeekLow": ticker.info["fiftyTwoWeekLow"], 
            "fiftyTwoWeekHigh": ticker.info["fiftyTwoWeekHigh"], 
            "fiftyTwoWeekRange": ticker.info["fiftyTwoWeekRange"],
            "fiftyDayAverage": ticker.info["fiftyDayAverage"],
            "twoHundredDayAverage": ticker.info["twoHundredDayAverage"],
            "fiftyTwoWeekLowChange": ticker.info["fiftyTwoWeekLowChange"],
            "fiftyTwoWeekLowChangePercent": ticker.info["fiftyTwoWeekLowChangePercent"],
            "fiftyTwoWeekHighChange": ticker.info["fiftyTwoWeekHighChange"],
            "fiftyTwoWeekHighChangePercent": ticker.info["fiftyTwoWeekHighChangePercent"],
            "fiftyTwoWeekChangePercent": ticker.info["fiftyTwoWeekChangePercent"],
            "volume": ticker.info["volume"],
            "regularMarketVolume": ticker.info["regularMarketVolume"],
            "averageVolume": ticker.info["averageVolume"],
            "averageVolume10days": ticker.info["averageVolume10days"],
            "averageDailyVolume10Day": ticker.info["averageDailyVolume10Day"],
            "averageDailyVolume3Month": ticker.info["averageDailyVolume3Month"],
            "bid": ticker.info["bid"],
            "ask": ticker.info["ask"],
            "bidSize": ticker.info["bidSize"],
            "askSize": ticker.info["askSize"],
            "regularMarketChange": ticker.info["regularMarketChange"],
            "regularMarketChangePercent": ticker.info["regularMarketChangePercent"],
            "SandP52WeekChange": ticker.info["SandP52WeekChange"],
            "targetHighPrice": ticker.info["targetHighPrice"],
            "targetLowPrice": ticker.info["targetLowPrice"],
            "targetMeanPrice": ticker.info["targetMeanPrice"],
            "targetMedianPrice": ticker.info["targetMedianPrice"],
            "recommendationMean": ticker.info["recommendationMean"],
            "recommendationKey": ticker.info["recommendationKey"],
            "averageAnalystRating": ticker.info["averageAnalystRating"],
            "numberOfAnalystOpinions": ticker.info["numberOfAnalystOpinions"],
            "lastSplitFactor": ticker.info["lastSplitFactor"],
            "lastSplitDate": ticker.info["lastSplitDate"],
            "corporateActions": ticker.info["corporateActions"],
            "beta": ticker.info["beta"],
            "updateAt": current_date_time,
        }

        technicals_df = pd.DataFrame([technicals_data])
        logger.info(f"Successfully fetched technicals data for {ticker.info['longName']}.")
        return technicals_df
    except Exception as e:
        logger.error(f"Error fetching technicals data: {e}")
        raise ValueError(f"No data returned for '{ticker.info['longName']}'.")

async def extract(start_date:str, end_date:str):
    """Run all extractors concurrently and return their results.

    Args:
        start_date (str): Start date in ISO format YYYY-MM-DD for historical data.
        end_date (str): End date in ISO format YYYY-MM-DD for historical data (exclusive in yfinance).

    Returns:
        tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
            A tuple of (stock_df, company_profile_df, fundamentals_df, technicals_df).
    """
    ticker = yf.Ticker(STOCK_NAME)
    stock_df, company_profile_df, fundamentals_df, technicals_df = await asyncio.gather(
        get_stock_data(ticker, start_date, end_date),
        get_company_profile(ticker),
        get_fundamentals_data(ticker),
        get_technicals_data(ticker)
    )
    return stock_df, company_profile_df, fundamentals_df, technicals_df

if __name__ == "__main__":
    stock_df, company_profile_df, fundamentals_df, technicals_df = asyncio.run(extract(start_date="2025-08-01", end_date="2025-08-09"))
    print(stock_df.head())
    print(company_profile_df.head())
    print(fundamentals_df.head())
    print(technicals_df.head())