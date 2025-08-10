"""
This module contains the tasks for training the models.

The tasks are:
- train_regression_model: Train a regression model
- finetune_regression_model: Finetune the regression models
- train_lstm_model: Train an LSTM model
- finetune_lstm_model: Finetune the LSTM models
- main_training: Main training task that runs the finetuning tasks for both regression and LSTM models
"""

import os
import uuid
import pandas as pd
import numpy as np
import asyncio
import mlflow
from prefect import task, get_run_logger
from mlflow.models import infer_signature
from helper import *
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor



mlflow.set_tracking_uri(f"https://dagshub.com/{USERNAME}/{REPO_NAME}.mlflow")

def get_regression_model_params():
    """
    Get the parameters for the regression models
    """
    models_params = {
        "LinearRegression": {
            "model": LinearRegression,
            "params": [{}]  # no hyperparams to tune
        },
        "Ridge": {
            "model": Ridge,
            "params": [{"alpha": a} for a in [0.1, 1.0, 10.0]]
        },
        "Lasso": {
            "model": Lasso,
            "params": [{"alpha": a} for a in [0.001, 0.01, 0.1]]
        },
        "RandomForest": {
            "model": RandomForestRegressor,
            "params": [{"n_estimators": n, "max_depth": d} for n in [50, 100] for d in [5, 10]]
        },
        "GradientBoosting": {
            "model": GradientBoostingRegressor,
            "params": [{"n_estimators": n, "learning_rate": lr} for n in [50, 100] for lr in [0.05, 0.1]]
        }
    }
    return models_params

@task(name="train_regression_model")
async def train_regression_model(model_name, model_class, params, X_train, y_train, X_test, y_test):
    """
    Train a regression model
    Args:
        model_name: Name of the model
        model_class: Class of the model
        params: Parameters for the model
        X_train: Training data
        y_train: Training labels
        X_test: Testing data
        y_test: Testing labels
    """
    logger = get_run_logger()
    
    logger.info(f"Training {model_name} with params: {params}")
    
    uuid_str = str(uuid.uuid4())
    with mlflow.start_run(run_name=f"{model_name}_{uuid_str}", tags={"model_type": "regression"}):
        model = model_class(**params)
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Log params
        mlflow.log_params(params)
    
        # Log metrics
        mlflow.log_metric("mse_train", mean_squared_error(y_train, y_train_pred))
        mlflow.log_metric("rmse_train", np.sqrt(mean_squared_error(y_train, y_train_pred)))
        mlflow.log_metric("mae_train", mean_absolute_error(y_train, y_train_pred))
        mlflow.log_metric("mape_train", mean_absolute_percentage_error(y_train, y_train_pred))
        mlflow.log_metric("r2_train", r2_score(y_train, y_train_pred))
        mlflow.log_metric("mse_test", mean_squared_error(y_test, y_test_pred))
        mlflow.log_metric("rmse_test", np.sqrt(mean_squared_error(y_test, y_test_pred)))
        mlflow.log_metric("mae_test", mean_absolute_error(y_test, y_test_pred))
        mlflow.log_metric("mape_test", mean_absolute_percentage_error(y_test, y_test_pred))
        mlflow.log_metric("r2_test", r2_score(y_test, y_test_pred))

        # Create signature & input example
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train[:5]
        
        # Log model
        mlflow.sklearn.log_model(model, "model", signature=signature, input_example=input_example)
    logger.info(f"Trained {model_name} with params: {params}")

@task(name="finetune_regression_model")
async def finetune_regression_model(X_train, y_train, X_test, y_test):
    """
    Finetune the regression models
    Args:
        X_train: Training data
        y_train: Training labels
        X_test: Testing data
        y_test: Testing labels
    """
    logger = get_run_logger()
    model_params = get_regression_model_params()
    
    logger.info(f"Finetuning regression model with params: {model_params}")
    for model_name, mp in model_params.items():
        for params in mp["params"]:
            logger.info(f"Finetuning {model_name} with params: {params}")
            await train_regression_model(model_name, mp["model"], params, X_train, y_train, X_test, y_test)
            
    logger.info("Finetuned regression models")

def get_params_for_lstm():
    """
    Get the parameters for the LSTM models
    """
    lstm_params_list = [
        {"units": 32, "dropout": 0.2},
        {"units": 32, "dropout": 0.3},
        {"units": 64, "dropout": 0.3},
        {"units": 64, "dropout": 0.2},
    ]
    return lstm_params_list

@task(name="train_lstm_model")
async def train_lstm_model(input_shape, X_train, y_train, X_test, y_test, params):
    """
    Train an LSTM model
    Args:
        input_shape: Shape of the input data
        X_train: Training data
        y_train: Training labels
        X_test: Testing data
        y_test: Testing labels
        params: Parameters for the model
    """
    logger = get_run_logger()
    run_id = str(uuid.uuid4())
    logger.info(f"Training LSTM model with params: {params}")
    with mlflow.start_run(run_name=f"LSTM_{run_id}", tags={"model_type": "lstm"}):
        # Build LSTM model
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(params["units"], input_shape=input_shape, return_sequences=True),
            tf.keras.layers.Dropout(params["dropout"]),
            tf.keras.layers.LSTM(units=params["units"]),
            tf.keras.layers.Dropout(params["dropout"]),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=16,
            verbose=0
        )
        
        # Predictions
        y_train_pred = model.predict(X_train).flatten()
        y_test_pred = model.predict(X_test).flatten()
        
        # Log params
        mlflow.log_params(params)
        
        # Log metrics (train & test)
        mlflow.log_metric("mse_train", mean_squared_error(y_train, y_train_pred))
        mlflow.log_metric("rmse_train", np.sqrt(mean_squared_error(y_train, y_train_pred)))
        mlflow.log_metric("mae_train", mean_absolute_error(y_train, y_train_pred))
        mlflow.log_metric("mape_train", mean_absolute_percentage_error(y_train, y_train_pred))
        mlflow.log_metric("r2_train", r2_score(y_train, y_train_pred))
        
        mlflow.log_metric("mse_test", mean_squared_error(y_test, y_test_pred))
        mlflow.log_metric("rmse_test", np.sqrt(mean_squared_error(y_test, y_test_pred)))
        mlflow.log_metric("mae_test", mean_absolute_error(y_test, y_test_pred))
        mlflow.log_metric("mape_test", mean_absolute_percentage_error(y_test, y_test_pred))
        mlflow.log_metric("r2_test", r2_score(y_test, y_test_pred))
        
        # Log training history metrics
        for epoch, loss in enumerate(history.history["loss"]):
            mlflow.log_metric("loss", loss, step=epoch)
        for epoch, val_loss in enumerate(history.history["val_loss"]):
            mlflow.log_metric("val_loss", val_loss, step=epoch)
        
        for epoch, loss in enumerate(history.history["mae"]):
            mlflow.log_metric("mae", loss, step=epoch)
        for epoch, val_mae in enumerate(history.history["val_mae"]):
            mlflow.log_metric("val_mae", val_mae, step=epoch)
        
        # Signature & example input
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train[:5]
        
        # Log TensorFlow model
        mlflow.tensorflow.log_model(model, artifact_path="model", signature=signature, input_example=input_example)
    logger.info(f"Trained LSTM model with params: {params}")

@task(name="finetune_lstm_model")
async def finetune_lstm_model(X_train, y_train, X_test, y_test):
    """
    Finetune the LSTM models
    Args:
        X_train: Training data
        y_train: Training labels
        X_test: Testing data
        y_test: Testing labels
    """
    logger = get_run_logger()
    
    X_train_lstm = np.expand_dims(X_train, axis=1)
    X_test_lstm = np.expand_dims(X_test, axis=1)

    lstm_params_list = get_params_for_lstm()
    for params in lstm_params_list:
        logger.info(f"Finetuning LSTM model with params: {params}")
        await train_lstm_model(
            (X_train_lstm.shape[1], X_train_lstm.shape[2]),
            X_train_lstm,
            y_train,
            X_test_lstm,
            y_test,
            params
        )
    logger.info("Finetuned LSTM models")

@task(name="main_training")
async def main_training(training_set_regression, test_set_regression, training_set_lstm, test_set_lstm):
    """
    Main training task
    Args:
        training_set_regression: Training data for regression
        test_set_regression: Testing data for regression
        training_set_lstm: Training data for LSTM
        test_set_lstm: Testing data for LSTM
    """
    logger = get_run_logger()
    logger.info("Starting main training")
    await asyncio.gather(
        finetune_regression_model(training_set_regression[0], training_set_regression[1], test_set_regression[0], test_set_regression[1]),
        finetune_lstm_model(training_set_lstm[0], training_set_lstm[1], test_set_lstm[0], test_set_lstm[1])
    )
    logger.info("Main training completed")

if __name__ == "__main__":
    from data_ingestion import data_ingestion
    from data_preprocessing import main_data_preprocessing
    
    df = asyncio.run(data_ingestion())
    training_set_regression, training_set_lstm, test_set_regression, test_set_lstm = asyncio.run(main_data_preprocessing(df))
    asyncio.run(main_training(training_set_regression, test_set_regression, training_set_lstm, test_set_lstm))
