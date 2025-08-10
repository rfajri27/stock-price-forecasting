"""
This module contains the tasks for evaluating the model.

The tasks are:
- get_best_run: Get the best run from the experiment
"""

from helper import *
from mlflow.tracking import MlflowClient
import pandas as pd
import mlflow
from prefect import task, get_run_logger

@task(name="get_best_run")
async def get_best_run():
    """
    Get the best run from the experiment
    """
    logger = get_run_logger()
    logger.info("Getting best run from the experiment")
    experiment = client.get_experiment_by_name("Default")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    best_run = runs.sort_values(by=["metrics.mape_test", "metrics.r2_test"], ascending=[True, False]).iloc[0]
    
    logger.info(f"Best run ID: {best_run['run_id']}")
    logger.info(f"Model type: {best_run['tags.model_type']}")
    logger.info(f"Model name: {best_run['tags.mlflow.runName']}")
    logger.info(f"MAPE: {best_run['metrics.mape_test']}")
    logger.info(f"R²: {best_run['metrics.r2_test']}")
    
    return best_run

if __name__ == "__main__":
    import asyncio
    best_run = asyncio.run(get_best_run())
    print(best_run)