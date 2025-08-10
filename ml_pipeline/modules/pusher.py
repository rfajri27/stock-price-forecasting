from modules.helper import *
from mlflow.tracking import MlflowClient
import pandas as pd
import mlflow
from prefect import task, get_run_logger

@task(name="push_model_to_registry")
async def push_model_to_registry(model_uri):
    """
    Push the model to the registry
    """
    logger = get_run_logger()
    
    model_name = "stock_price_forecasting"
    logger.info("Pushing model to the registry")
    
    model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
    
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="production",
        archive_existing_versions=True
    )
    logger.info(f"Model {model_name} v{model_version.version} is now in Production.")
    
    return model_version

if __name__ == "__main__":
    import asyncio
    model_uri = "runs:/161a7cdf296948fa9e7d4badb981973e/model"
    model_version = asyncio.run(push_model_to_registry(model_uri))
    print(model_version)