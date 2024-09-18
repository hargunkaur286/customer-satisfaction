import logging
import pandas as pd
from zenml import step
from sklearn.linear_model import LinearRegression  # Import LinearRegression
from sklearn.base import RegressorMixin
from .config import ModelNameConfig
from zenml.client import Client
import mlflow

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig,
) -> RegressorMixin:
    """
    Trains the model on the ingested Data

    Args:
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame
        config: ModelNameConfig (defines which model to use)
    """
    try:
        model = None
        # Check the model_name from config and instantiate the appropriate model
        if config.model_name == "LinearRegressionModel":
            mlflow.sklearn.autolog()
            model = LinearRegression() 
            model.fit(X_train, y_train)  
            logging.info("LinearRegression model trained successfully.")
            return model
        else:
            raise ValueError(f"Model {config.model_name} not supported")
    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise e