import logging
import pandas as pd
from zenml import step
from typing import Tuple
from typing_extensions import Annotated
from sklearn.base import RegressorMixin
from zenml.client import Client
from src.evaluation import MSE, R2, RMSE
import mlflow 

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> Tuple[
    Annotated[float, "r2_score"],
    Annotated[float, "rmse"],
]:
    """
    Evaluates the model on the test data.

    Args:
        model: Trained model to evaluate
        X_test: Features for testing
        y_test: True labels for testing

    Returns:
        r2_score: Coefficient of determination (R^2)
        rmse: Root Mean Squared Error
    """
    try:
        # Get predictions
        predictions = model.predict(X_test)

        # Calculate metrics
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, predictions)
        mlflow.log_metric("mse", mse)

        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test, predictions)
        mlflow.log_metric("r2", r2)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, predictions)
        mlflow.log_metric("rmse", rmse)

        logging.info(f"Evaluation completed. R2 score: {r2}, RMSE: {rmse}")
        
        # Return R2 and RMSE as floats
        return float(r2), float(rmse)

    except Exception as e:
        logging.error(f"Error in evaluating model: {e}")
        raise e
