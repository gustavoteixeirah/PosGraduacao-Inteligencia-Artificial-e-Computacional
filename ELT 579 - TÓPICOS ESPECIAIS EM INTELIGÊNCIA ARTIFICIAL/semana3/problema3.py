import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.client import device_lib

# %% Load dataset
df = pd.read_csv('dataset.csv')

df.head()
df.shape

# %% Dados faltantes

for column_name in df.columns:
    if df[column_name].isna().sum() != 0:
        print(column_name, df[column_name].isna().sum())

df = df.fillna(0)

# %%
X = df.drop(['ID', 'produtividade'], axis=1)
y = df.loc[:, ['produtividade']]

# %% dividir o banco de dados em treinamento e teste.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)

# %%
from sklearn.preprocessing import StandardScaler

scaleX = StandardScaler()  # cria um objeto
scaleX = scaleX.fit(X_train)  # ajusta aos dados
X_train_sc = scaleX.transform(X_train)  # transforma os dados
X_test_sc = scaleX.transform(X_test)  # utransforma os dados

X_train_sc = pd.DataFrame(X_train_sc)
X_train_sc.columns = X_train.columns

X_test_sc = pd.DataFrame(X_test_sc)
X_test_sc.columns = X_train.columns

# %% Redução de dimensionalidade
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

X_train_pca = pca.fit_transform(X_train_sc)
X_test_pca = pca.transform(X_test_sc)

# %% Modelo
# Contruir o modelo de rede neural artificial totalmente conectada
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

ann = Sequential()

ann.add(Dense(units=5, input_dim=2, activation='relu'))
ann.add(Dense(units=4, activation='relu'))
ann.add(Dense(units=1))

# %%  compile
ann.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
ann.summary()

# %% Treinamento da rede neural

history = ann.fit(X_train_pca, y_train, batch_size=10,
                  validation_split=0.1, epochs=600)

# %%
# plotar gráficos
import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'y', label="Treinamento")
plt.plot(epochs, val_loss, 'r', label="Validação")

plt.title("Treinamento versus validação")
plt.xlabel("Epocas")
plt.ylabel("Função custo (Loss)")
plt.legend()
plt.show()

# %% Teste

y_pred = ann.predict(X_test_pca)

plt.scatter(y_test, y_pred)
plt.show()

# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

linear = LinearRegression()
linear.fit(y_test, y_pred)

r2 = linear.score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print('R2 = ', round(r2, 2))
print("MAE = ", round(mae, 2))
print("MSE = ", round(mse, 2))

# predição dos dados de teste
y_pred = ann.predict(X_test_pca)

y = np.arange(0, 1200)
y = np.expand_dims(y, axis=1)

yp = linear.predict(y)

plt.scatter(y_test, y_pred)
plt.plot(y, yp)
plt.show()

# %%
# R2 =  0.09
# MAE =  231.66
# MSE =  78265.3

# R2 =  0.16
# MAE =  198.72
# MSE =  59801.7

# R2 =  0.28
# MAE =  129.54
# MSE =  30119.65
