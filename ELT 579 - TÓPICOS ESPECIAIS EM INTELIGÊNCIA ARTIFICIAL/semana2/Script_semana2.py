# -*- coding: utf-8 -*-
"""

@author: sarvio valente
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#%% importar o dataset

df = pd.read_csv('dataset_tomate_com_severidade.csv')

X = df.drop(['id', 'Severidade'], axis = 1)
y = df['Severidade']

#%% separar dados de treinamento e teste

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#%% Padronização das variáveis

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler() #media 0 e desvio padrão 1

X_train_sc = scaler.fit_transform(X_train)

X_test_sc = scaler.transform(X_test)


X_train_sc = pd.DataFrame(X_train_sc)
X_train_sc.columns = X_train.columns

X_test_sc = pd.DataFrame(X_test_sc)
X_test_sc.columns = X_train.columns

#%% validação cruzada para selecionar o número de variáveis

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

modelo_linear = LinearRegression()

score = cross_val_score(modelo_linear, X_train_sc, y_train, cv = 10)

print(np.mean(score))


#%% seleção de variáveis

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

max_f = 20

lista_r2 = list()

for i in range(1, max_f+1):
  
  selector = RFE(modelo_linear, n_features_to_select=i, step=1)
  
  selector = selector.fit(X_train_sc, y_train)
  
  mask = selector.support_
  
  features = X_train_sc.columns
  
  sel_features = features[mask]

  X_sel = X_train_sc[sel_features]
  
  modelo_linear = LinearRegression()
  
  score = cross_val_score(modelo_linear, X_sel, y_train, cv = 10, scoring = 'neg_root_mean_squared_error')
  
  lista_r2.append(np.abs(np.mean(score)))
  
  print(np.abs(np.mean(score)))

#%% gráfico

import matplotlib.pyplot as plt

plt.plot(lista_r2)

plt.show()

#%% selecionar as features

modelo_linear = LinearRegression()

selector = RFE(modelo_linear, n_features_to_select = 10, step=1)

selector = selector.fit(X_train_sc, y_train)

mask = selector.support_

features = X_train_sc.columns

sel_features = features[mask]

print(sel_features)


#%% validação cruzada

modelo_linear = LinearRegression()

X_sel = X_train_sc[sel_features]

score = cross_val_score(modelo_linear, X_sel, y_train, cv = 10, scoring = 'r2')

print(np.mean(score))

#%% modelo final - Regressão linear multipla

from sklearn.linear_model import LinearRegression

modelo_linear = LinearRegression()

modelo_linear.fit(X_sel, y_train)

coef = modelo_linear.coef_

print(coef)

r2 = modelo_linear.score(X_train, y_train)


#%% teste final
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


y_pred = modelo_linear.predict(X_test_sc[features])


r2 = modelo_linear.score(X_test_sc[features], y_test)

rmse = (mean_squared_error(y_test, y_pred))**0.5

mae = mean_absolute_error(y_test, y_pred)

print("RMSE", rmse)
print("MAE", mae)
print("R2", r2)


plt.scatter(y_test, y_pred)

plt.show()

#%% R2 da validação


model_r2 = LinearRegression()
model_r2.fit(np.array(y_test).reshape(-1, 1), y_pred)

import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred)

print(model_r2.score(np.array(y_test).reshape(-1, 1), y_pred))














