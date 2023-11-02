import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# %% IMPORT DATASET

df = pd.read_csv('dataset.csv')

X = df.drop(['id', 'Severidade'], axis=1)

y = df['Severidade']

# %% Split dos dados
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# %% Padronizacao dos dados
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# %% Validacao cruzada
modelo_linear = LinearRegression()
score = cross_val_score(modelo_linear, X_train, y_train, cv=10, scoring='r2')

np.mean(score)

# %% Modelo final
modelo_linear = LinearRegression()
modelo_linear.fit(X_train_sc, y_train)

#%% Visualizacao dos coeficientes ( importancia das features)
coef = modelo_linear.coef_

print(coef)

features = pd.DataFrame(coef)
features['features'] = X_train.columns

# %%
