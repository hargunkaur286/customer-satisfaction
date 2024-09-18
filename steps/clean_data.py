import logging
import pandas as pd
from zenml import step 

from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreProcessStrategy
from typing_extensions import Annotated
from typing import Tuple

@step 
def clean_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.DataFrame, "y_train"],
    Annotated[pd.DataFrame, "y_test"],
]:
    """
    Cleans the data and divides it into training and testing

    Args:
        df: Raw data
    Returns:
        X_train: Training Data
        X_test: Testing Data
        y_train: Training labels
        y_test: Testing labels
    """
    try:
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()

        if isinstance(y_train, pd.Series):
            y_train = pd.DataFrame(y_train)
        if isinstance(y_test, pd.Series):
            y_test = pd.DataFrame(y_test)

        logging.info("Data cleaning and division completed successfully")
        return X_train, X_test, y_train, y_test

    except Exception as e:
        logging.error("Error in cleaning data: {}".format(e))
        raise e
