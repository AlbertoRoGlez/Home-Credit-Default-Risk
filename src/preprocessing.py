from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarrary
        val : np.ndarrary
        test : np.ndarrary
    """
    # Print shape of input data
    print("Input train data shape: ", train_df.shape)
    print("Input val data shape: ", val_df.shape)
    print("Input test data shape: ", test_df.shape, "\n")

    # Make a copy of the dataframes
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    # 1. Correct outliers/anomalous values in numerical
    # columns (`DAYS_EMPLOYED` column).
    working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

    # 2. TODO Encode string categorical features (dytpe `object`):
    #     - If the feature has 2 categories encode using binary encoding,
    #       please use `sklearn.preprocessing.OrdinalEncoder()`. Only 4 columns
    #       from the dataset should have 2 categories.
    #     - If it has more than 2 categories, use one-hot encoding, please use
    #       `sklearn.preprocessing.OneHotEncoder()`. 12 columns
    #       from the dataset should have more than 2 categories.
    # Take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the OrdinalEncoder and
    #     OneHotEncoder classes, then use the fitted models to transform all the
    #     datasets.

    # Encode categorical features
    ordinal_cols = [col for col in train_df.columns if train_df[col].nunique() == 2 and train_df[col].dtype == "object"]
    onehot_cols = [col for col in train_df.columns if col not in ordinal_cols and train_df[col].dtype == "object"]
    ordinal_encoder = OrdinalEncoder()
    onehot_encoder = OneHotEncoder()
    ct = ColumnTransformer([
        ("ordinal", ordinal_encoder, ordinal_cols),
        ("onehot", onehot_encoder, onehot_cols),
    ], remainder="passthrough")

    # Fit and transform on the training data
    working_train_df = ct.fit_transform(working_train_df)

    # Transform the validation and test data using the fitted transformer
    working_val_df = ct.transform(working_val_df)
    working_test_df = ct.transform(working_test_df)

    # 3. TODO Impute values for all columns with missing data or nan, just all the columns.
    # Use median as imputing value. Please use sklearn.impute.SimpleImputer().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the SimpleImputer and then use the fitted
    #     model to transform all the datasets.

    imputer = SimpleImputer(strategy="median")
    working_train_df = imputer.fit_transform(working_train_df)
    working_val_df = imputer.transform(working_val_df)
    working_test_df = imputer.transform(working_test_df)

    # 4. TODO Feature scaling with Min-Max scaler. Apply this to all the columns.
    # Please use sklearn.preprocessing.MinMaxScaler().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the MinMaxScaler and then use the fitted
    #     model to transform all the datasets.

    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(working_train_df)
    working_train_df = scaler.transform(working_train_df)
    working_val_df = scaler.transform(working_val_df)
    working_test_df = scaler.transform(working_test_df)

    return working_train_df, working_val_df, working_test_df