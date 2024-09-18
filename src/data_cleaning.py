import logging
from abc import ABC, abstractmethod
from typing import Union, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    """
    Abstract class defining strategy for handling data
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreProcessStrategy(DataStrategy):
    """
    Strategy for preprocessing data
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess Data
        """
        try:
            # Dropping unnecessary columns
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],
                axis=1)
            
            # Handling null values
            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)
            data["review_comment_message"].fillna("No Review", inplace=True)
            
            # Selecting only numerical data
            data = data.select_dtypes(include=[np.number])
            
            # Dropping additional columns
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)

            return data
        except Exception as e:
            logging.error(f"Error in preprocessing data: {e}")
            raise e


class DataDivideStrategy(DataStrategy):
    """
    Strategy for dividing data into training and testing.
    """
    def handle_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        try:
            X = data.drop(["review_score"], axis=1)
            y = data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error in dividing data: {e}")
            raise e


class DataCleaning:
    """
    Class for cleaning data which processes the data and divides it into train and test
    """
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series, Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        """Handle Data"""
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in handling data: {e}")
            raise e


if __name__ == "__main__":
    # Example Usage
    data = pd.read_csv("data/olist_customers_dataset.csv")
    
    # Preprocess the data
    data_cleaning = DataCleaning(data, DataPreProcessStrategy())
    processed_data = data_cleaning.handle_data()
    
    # Divide the data into train and test sets
    data_cleaning = DataCleaning(processed_data, DataDivideStrategy())
    X_train, X_test, y_train, y_test = data_cleaning.handle_data()
