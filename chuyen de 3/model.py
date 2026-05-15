import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from prophet import Prophet


# =====================================================
# EVALUATE FUNCTION
# =====================================================
def evaluate_model(y_true, y_pred):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    mape = np.mean(
        np.abs(
            (y_true - y_pred) / y_true
        )
    ) * 100

    return mae, rmse, r2, mape


# =====================================================
# LINEAR REGRESSION
# =====================================================
def linear_model(X_train, X_test, y_train):

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(
        X_test
    )

    return pred


# =====================================================
# RANDOM FOREST
# =====================================================
def random_forest_model(
    X_train,
    X_test,
    y_train
):

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(
        X_test
    )

    return pred


# =====================================================
# PROPHET
# =====================================================
def prophet_model(df):

    prophet_df = pd.DataFrame()

    prophet_df["ds"] = df["Order Date"]

    prophet_df["y"] = df["Sales"]

    model = Prophet()

    model.fit(prophet_df)

    future = model.make_future_dataframe(
        periods=30
    )

    forecast = model.predict(future)

    return forecast
