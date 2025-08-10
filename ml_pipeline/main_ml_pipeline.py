from prefect import flow, get_run_logger

from modules.helper import *
from modules.data_ingestion import data_ingestion
from modules.data_preprocessing import main_data_preprocessing
from modules.training import main_training
from modules.evaluation import get_best_run
from modules.pusher import push_model_to_registry

@flow(name="main_ml_pipeline")
async def main_ml_pipeline():
    logger = get_run_logger()
    logger.info("Starting the ML pipeline")
    
    df = await data_ingestion()
    training_set_regression, training_set_lstm, test_set_regression, test_set_lstm = await main_data_preprocessing(df)
    await main_training(
        training_set_regression, 
        test_set_regression, 
        training_set_lstm, 
        test_set_lstm
    )
    best_run = await get_best_run()
    model_uri = f"runs:/{best_run['run_id']}/model"
    model_version = await push_model_to_registry(model_uri)
    logger.info(f"Model version: {model_version}")
    return model_version

if __name__ == "__main__":
    import asyncio
    asyncio.run(main_ml_pipeline())