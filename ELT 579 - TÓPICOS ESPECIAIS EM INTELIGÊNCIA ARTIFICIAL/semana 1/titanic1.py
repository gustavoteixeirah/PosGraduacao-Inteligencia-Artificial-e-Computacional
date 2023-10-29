#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 21:09:47 2023

@author: teixeira
"""


import pandas as pd
import numpy as np

# baseline 76%

#%% criar dados de treino e de teste

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#%% Pre processamento dos dados

# descricao estatisticas das features numericas
est = train.describe()
print(train.info())

# verificar valores nulos ou NAN
print(train.isnull().sum())
print(test.isnull().sum())

#%% Mapear colunas
col = pd.Series(list(train.columns))


X_train = train.drop(['PassengerId', 'Survived'], axis = 1)
X_test = test.drop(['PassengerId'], axis = 1)

#%% Criar feature

subs = {'female':1, 'male':0}
X_train['mulher'] = X_train['Sex'].replace(subs)
X_test['mulher'] = X_test['Sex'].replace(subs)

#%% Selecionar as features

X_train = X_train[['mulher']]
X_test = X_test[['mulher']]

y_train = train['Survived']

#%% Modelo e validacao cruzada

# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
# from sklearn.metrics import confusion_matrix
# from sklearn.preprocessing import StandardScaler
# pip install scikit-learn
model_lr = LogisticRegression()
score = cross_val_score(model_lr, X_train, y_train, cv = 10)

print(np.mean(score))

#%% Modelo final

model_lr.fit(X_train, y_train)

y_pred = model_lr.predict(X_train)

#%% matrix de confusao
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_train, y_pred)

print(cm)

score = model_lr.score(X_train, y_train)
print(score)

#%% Previsao final

y_pred = model_lr.predict(X_test)

#%% Criar arquivo de submissao

submission = pd.DataFrame(test['PassengerId'])
submission['Survived'] = y_pred

submission.to_csv('submission1.csv', index = False)











