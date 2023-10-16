from tensorflow import keras
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

fashion_mnist = keras.datasets.fashion_mnist

(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# Divisão em 80% treinamento e 20% validação
split = int(0.8 * len(X_train_full))
X_train, X_valid = X_train_full[:split] / 255., X_train_full[split:] / 255.
y_train, y_valid = y_train_full[:split], y_train_full[split:]

X_test = X_test / 255.

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.BatchNormalization(),  # Adicionando normalização
    keras.layers.Dense(500, activation="relu"),  # Aumentando o número de unidades
    keras.layers.Dropout(0.2),  # Adicionando dropout
    keras.layers.Dense(300, activation="relu"),  # Aumentando o número de unidades
    keras.layers.Dropout(0.2),  # Adicionando dropout
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',  # Usando o otimizador 'adam'
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

model.evaluate(X_test, y_test)
