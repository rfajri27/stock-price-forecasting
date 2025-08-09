import os
import logging
import dotenv
from datetime import datetime
import pytz
from sqlalchemy import create_engine

dotenv.load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

jakarta_timezone = pytz.timezone("Asia/Jakarta")
current_date = datetime.now(jakarta_timezone).strftime("%Y-%m-%d")
current_date_time = datetime.now(jakarta_timezone).strftime("%Y-%m-%d %H:%M:%S")
STOCK_NAME = "BBCA.JK"

def get_db_engine():
    """Get a connection to the database."""
    engine = create_engine(os.getenv("DB_URL"))
    return engine