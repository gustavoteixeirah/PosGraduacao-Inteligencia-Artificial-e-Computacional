import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

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

# %% Selecao de Features
from sklearn.feature_selection import RFE

max_features = 20
score_list = list()
for i in range(1, max_features + 1):
    modelo_linear = LinearRegression()

    selector = RFE(modelo_linear, n_features_to_select=i, step=1)
    selector = selector.fit(X_train_sc, y_train)

    mask = selector.support_

    features = X_train_sc.columns
    sel_features = features[mask]

    X_sel = X_train_sc[sel_features]

    score_sel = cross_val_score(modelo_linear, X_sel, y_train, cv=10, scoring="r2")
    mean_score = np.mean(score_sel)
    print("i: " + str(i) + " = " + str(mean_score))
    score_list.append(score_sel)

# %% Plot dos scores
import matplotlib.pyplot as plt

plt.plot(score_list)
plt.show()

# %% Selecao final de features

modelo_linear = LinearRegression()

selector = RFE(modelo_linear, n_features_to_select=10, step=1)
selector = selector.fit(X_train_sc, y_train)

mask = selector.support_

features = X_train_sc.columns
sel_features = features[mask]

X_sel = X_train_sc[sel_features]

score_sel = cross_val_score(modelo_linear, X_sel, y_train, cv=10, scoring="r2")
print(np.mean(score_sel))
print(sel_features)


# %% Validacao cruzada
modelo_linear = LinearRegression()
score = cross_val_score(modelo_linear, X_sel, y_train, cv=10, scoring="r2")

np.mean(score)

# %% Modelo final
modelo_linear = LinearRegression()
modelo_linear.fit(X_sel, y_train)

# %% Visualizacao dos coeficientes ( importancia das features)
coef = modelo_linear.coef_

print(coef)

features = pd.DataFrame(coef)
features["features"] = X_train.columns

# %% testar nos dados de teste

y_pred = modelo_linear.predict(X_test_sc[sel_features])

r2 = modelo_linear.score(X_test_sc[sel_features], y_test)

print("r2", r2)

# %% outras metricas
from sklearn.metrics import mean_squared_error, mean_absolute_error

rmse = mean_squared_error(y_test, y_pred, squared=False)

mae = mean_absolute_error(y_test, y_pred)

print("rmse", rmse)
print("mae", mae)
print("r2", r2)
# rmse 7.539476072286391
# mae 6.3440663489401565
# r2 0.8876342778701652
# %%
