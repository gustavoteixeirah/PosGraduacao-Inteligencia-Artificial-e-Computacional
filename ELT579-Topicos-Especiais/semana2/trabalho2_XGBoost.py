import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_selection import RFE

# %% IMPORT DATASET

df = pd.read_csv("dataset.csv")

X = df.drop(["id", "Severidade"], axis=1)
y = df["Severidade"]

# %% Split dos dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# %% Padronizacao dos dados
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)


# %% Usar XGBoost ao invés de Linear Regression

xgb = XGBRegressor(
    n_estimators=100,
    learning_rate=0.08,
    gamma=0,
    subsample=0.75,
    colsample_bytree=1,
    max_depth=7,
)

xgb.fit(X_train_sc, y_train)

predictions = xgb.predict(X_test_sc)

r2 = xgb.score(X_test_sc, y_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)

print("r2", r2)
print("rmse", rmse)
print("mae", mae)

# %%
