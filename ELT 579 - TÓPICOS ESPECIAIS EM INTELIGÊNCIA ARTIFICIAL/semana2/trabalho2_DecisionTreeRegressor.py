import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

# %% recuperar os nomes nos dataframes
X_train_sc = pd.DataFrame(X_train_sc, columns=X_train.columns)
X_test_sc = pd.DataFrame(X_test_sc, columns=X_train.columns)

# %% Definicao do modelo
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

# modelo = KNeighborsRegressor(n_neighbors=5, p=2, weights="distance")
modelo = RandomForestRegressor()

# %% Selecao de Features
from sklearn.feature_selection import RFE

max_features = 20
score_list = list()
for i in range(1, max_features + 1):
    modelo = RandomForestRegressor()
    selector = RFE(modelo, n_features_to_select=i, step=1)
    selector = selector.fit(X_train_sc, y_train)

    mask = selector.support_

    features = X_train_sc.columns
    sel_features = features[mask]

    X_sel = X_train_sc[sel_features]

    score_sel = cross_val_score(modelo, X_sel, y_train, cv=10, scoring="r2")
    mean_score = np.mean(score_sel)
    print("i: " + str(i) + " = " + str(mean_score))
    score_list.append(score_sel)

# %% gráfico

import matplotlib.pyplot as plt

plt.plot(score_list)

plt.show()

# %% selecionar as features


selector = RFE(modelo, n_features_to_select=11, step=1)

selector = selector.fit(X_train_sc, y_train)

mask = selector.support_

features = X_train_sc.columns

sel_features = features[mask]

print(sel_features)

# %% validação cruzada


X_sel = X_train_sc[sel_features]

score = cross_val_score(modelo, X_sel, y_train, cv=10, scoring="r2")

print(np.mean(score))

# %% modelo final


modelo.fit(X_sel, y_train)

coef = modelo.coef_

print(coef)

# r2 = modelo_linear.score(X_train, y_train)


# %% teste final
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


y_pred = modelo.predict(X_test_sc[features])


r2 = modelo.score(X_test_sc[features], y_test)

rmse = (mean_squared_error(y_test, y_pred)) ** 0.5

mae = mean_absolute_error(y_test, y_pred)

print("RMSE", rmse)
print("MAE", mae)
print("R2", r2)


plt.scatter(y_test, y_pred)

plt.show()

# %% R2 da validação


modelo.fit(np.array(y_test).reshape(-1, 1), y_pred)

import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)

print(modelo.score(np.array(y_test).reshape(-1, 1), y_pred))
