from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
import xgboost as xgb


def _encode_dates(X):
    
    X = X.copy()  
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour

    return X.drop(columns=["date"])

def _merge_external_data(X):
    df_ext = pd.read_csv("/kaggle/input/mdsb-2023/external_data.csv", parse_dates=["date"])

    X = X.copy()
    
    # Keeping the original index to sort back afterwards
    X["orig_index"] = np.arange(X.shape[0])
    X = pd.merge_asof(
        X.sort_values("date"), df_ext[["date", "ww", "u", "etat_sol"]].sort_values("date"), on="date" # Chosen weather-related features
    )
    # Sort back to the original order
    X = X.sort_values("orig_index")
    del X["orig_index"]
    return X

from workalendar.europe import France

def add_new_features(df):
    # Create an instance of the France calendar 
    cal = France()

    # Convert 'date' to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Adding relevant date-related features
    df['is_weekend'] = df['date'].dt.weekday.isin([5, 6]).astype(int)
    df['is_holiday'] = df['date'].apply(lambda x: cal.is_holiday(x)).astype(int)
        
    # Define curfew periods
    curfew_periods = [
        (pd.to_datetime("2020-10-17"), pd.to_datetime("2020-12-15"), 21, 6),
        (pd.to_datetime("2020-12-15"), pd.to_datetime("2021-01-16"), 20, 6),
        (pd.to_datetime("2021-01-16"), pd.to_datetime("2021-05-19"), 18, 6),
        (pd.to_datetime("2021-05-19"), pd.to_datetime("2021-06-09"), 21, 6),
        (pd.to_datetime("2021-06-09"), pd.to_datetime("2021-06-30"), 23, 6)
    ]

    # Function to check if a datetime is within the curfew period
    def is_curfew(date):
        hour = date.hour
        for start, end, start_hour, end_hour in curfew_periods:
            if start <= date <= end:
                if start_hour <= hour or hour < end_hour:  # Curfew hours
                    return 1
        return 0

    # Apply the function to each row
    df['is_curfew'] = df['date'].apply(is_curfew)
     
    # Adding cyclic encoding for day, month, and hour
    df['day_sin'] = np.sin(2 * np.pi * df['date'].dt.day / df['date'].dt.days_in_month)
    df['day_cos'] = np.cos(2 * np.pi * df['date'].dt.day / df['date'].dt.days_in_month)
    df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12)
    df['hour_sin'] = np.sin(2 * np.pi * df['date'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['date'].dt.hour / 24)

    return df

def create_pipeline():

    # Pipeline components
    date_encoder = FunctionTransformer(_encode_dates)
    date_cols = ["year", "month", "day", "weekday", "hour"]

    # Relevant features selected
    categorical_cols = [
        "counter_name", "site_name", "etat_sol", "ww", 
        "is_weekend", "is_holiday", "is_curfew"]
    numerical_cols = ["u", "day_sin", "day_cos", "month_sin", "month_cos", "hour_sin", "hour_cos"]
    
    preprocessor = ColumnTransformer(
        [
            ("date", OneHotEncoder(handle_unknown="ignore"), date_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", 'passthrough', numerical_cols) 
        ]
    )

    # XGB Regressor parameters
    regressor = xgb.XGBRegressor(max_depth=8, objective='reg:squarederror', learning_rate=0.2, n_estimators=100)

    # Final pipeline
    pipe = make_pipeline(
        FunctionTransformer(add_new_features, validate=False),
        FunctionTransformer(_merge_external_data, validate=False),
        date_encoder,
        preprocessor,
        regressor
    )

    return pipe

data_train = pd.read_parquet("/kaggle/input/mdsb-2023/train.parquet")
data_train['date'] = pd.to_datetime(data_train['date'])

# Filter out the data that falls within the lockdown period
lockdown_start = pd.to_datetime("2020-10-30")
lockdown_end = pd.to_datetime("2020-12-15")
data_train = data_train[~((data_train['date'] >= lockdown_start) & (data_train['date'] <= lockdown_end))]

X_train = data_train.drop(["bike_count", "log_bike_count"], axis=1)
y_train = data_train["log_bike_count"]

pipe = create_pipeline()
pipe.fit(X_train, y_train)

X_final_test = pd.read_parquet("/kaggle/input/mdsb-2023/final_test.parquet")
y_pred = pipe.predict(X_final_test)

# Replacing negative values by 0 (log(1+x) is always positive)
for i in range(len(y_pred)):
    if y_pred[i] < 0:
        y_pred[i] = 0

results = pd.DataFrame(
    dict(
        Id=np.arange(y_pred.shape[0]),
        log_bike_count=y_pred,
    )
)
results.to_csv("submission_XGB_curfew.csv", index=False)