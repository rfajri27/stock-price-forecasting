import os
import dotenv
from datetime import datetime
import pytz
from sqlalchemy import create_engine
import mlflow
from mlflow.tracking import MlflowClient
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

dotenv.load_dotenv()


jakarta_timezone = pytz.timezone("Asia/Jakarta")
current_date = datetime.now(jakarta_timezone).strftime("%Y-%m-%d")
current_date_time = datetime.now(jakarta_timezone).strftime("%Y-%m-%d %H:%M:%S")

USERNAME = "rfajri27"
REPO_NAME = "stock-price-forecasting"

os.environ['MLFLOW_TRACKING_USERNAME'] = USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("DAGSHUB_TOKEN")

mlflow.set_tracking_uri(f"https://dagshub.com/{USERNAME}/{REPO_NAME}.mlflow")
client = MlflowClient()

input_columns = ["Date", "Close", "lag_1", "lag_2", 
                  "lag_3", "lag_4", "lag_5", "rolling_mean_3"]
output_columns = ["target"]

def get_db_engine():
    """Get a connection to the database or None if DB_URL not set."""
    db_url = os.getenv("DB_URL")
    if not db_url:
        return None
    engine = create_engine(db_url)
    return engine