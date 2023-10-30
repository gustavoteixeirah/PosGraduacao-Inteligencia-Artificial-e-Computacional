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

def criar_features(X):
    subs = {'female':1, 'male':0}
    X['mulher'] = X['Sex'].replace(subs)
    
    X['Age'] = X['Age'].fillna(X['Age'].mean())
    
    X['Fare'] = X['Fare'].fillna(X['Fare'].mean())
    
    X['Embarked'] = X['Embarked'].fillna('S')
    
    subs = {'S':1, 'C':2, 'Q':3}
    X['porto'] = X['Embarked'].replace(subs)
    
    
    return X


X_train = criar_features(X_train)
X_test = criar_features(X_test)



#%% Selecionar as features


features = ['Pclass', 'Age', 'SibSp',
       'Parch', 'Fare', 'mulher', 'porto']
X_train = X_train[features]
X_test = X_test[features]

y_train = train['Survived']


#%% Padronizacao das variaveis

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


#%% Modelo e validacao cruzada

# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
# from sklearn.metrics import confusion_matrix
# from sklearn.preprocessing import StandardScaler
# pip install scikit-learn
model_lr = LogisticRegression(max_iter = 1000, random_state=0)
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











