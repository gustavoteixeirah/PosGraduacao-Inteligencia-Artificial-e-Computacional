#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 21:09:47 2023

@author: teixeira
"""
#%% start


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

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
import re

def criar_features(X):
    subs = {'female':1, 'male':0}
    X['mulher'] = X['Sex'].replace(subs)
    
    X['Age'] = X['Age'].fillna(X['Age'].mean())
    
    X['Fare'] = X['Fare'].fillna(X['Fare'].mean())
    
    X['Embarked'] = X['Embarked'].fillna('S')
    
    subs = {'S':1, 'C':2, 'Q':3}
    X['porto'] = X['Embarked'].replace(subs)
    
    X['ehCrianca'] = 1
    X['ehCrianca'] = np.where(X['Age'] < 15, 1, 0)
    
    X['ehIdoso'] = 0
    X['ehIdoso'] = np.where(X['Age'] > 55, 1, 0)

    X['temCabine'] = np.where(pd.notnull(X['Cabin']), 1, 0)

    X['contemApenasNumerosNoTicket'] = X['Ticket'].apply(lambda x: 1 if x.isdigit() else 0)

    X['contemLetrasNoTicket'] = X['Ticket'].apply(lambda x: 1 if re.search('[a-zA-Z]', x) else 0)

    X['tamanhoDoTicket'] = X['Ticket'].apply(lambda x: len(x))

    return X


X_train = criar_features(X_train)
X_test = criar_features(X_test)


#%% Selecionar as features


features = ['Pclass', 'Age', 'SibSp',
       'Parch', 'Fare', 'mulher', 'porto', 'ehCrianca', 'ehIdoso', 'temCabine',
       'contemApenasNumerosNoTicket', 'contemLetrasNoTicket',  'tamanhoDoTicket']
X_train = X_train[features]
X_test = X_test[features]

y_train = train['Survived']


#%% Padronizacao das variaveis
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

#%% Random Forest
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(n_estimators=80, max_depth=3, random_state=0)

score = cross_val_score(model_rf, X_train, y_train, cv = 10)

model_rf_score = np.mean(score)
<<<<<<< HEAD
print("Decision tree: ", model_rf_score)

#%% Otimização de hiperparametros
from skopt import gp_minimize

parametros =[('entropy', 'gini'), 
             (100, 1000),
             (3, 20),
             (2,10),
             (1, 10)]

def treinar_modelo(parametros):
    model_rf = RandomForestClassifier(
                                      criterion=parametros[0],
                                      n_estimators=parametros[1],
                                      max_depth=parametros[2],
                                      min_samples_split=parametros[3],
                                      min_samples_leaf=parametros[4],
                                      random_state=0, n_jobs = -1 )

    score = cross_val_score(model_rf, X_train, y_train, cv = 10)

    mean_score = np.mean(score)
    
    return -mean_score
otimos = gp_minimize(treinar_modelo, parametros, random_state=0, verbose=1, n_calls=50, n_random_starts=10, n_jobs = -1)


#%% Ensamble

from sklearn.ensemble import VotingClassifier

model_voting = VotingClassifier(estimators=[('LR', model_lr),
                                             ('KNN', model_knc),
                                             ('SVC', model_svc),
                                             ('RF', model_rf)],
                                voting='hard')

model_voting.fit(X_train, y_train)

score = cross_val_score(model_voting, X_train, y_train, cv = 10)

model_voting_score = np.mean(score)
print("Decision tree: ", model_voting_score)
=======
print(" Random Forest  : ", model_rf_score)
>>>>>>> 1b9742c5b96b7ad60ba181e1d33da02e59d1faa9


#%% Modelo final

model_rf.fit(X_train, y_train)

y_pred = model_rf.predict(X_train)

#%% matrix de confusao
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_train, y_pred)

print(cm)

score = model_rf.score(X_train, y_train)
print(score)

#%% Previsao final

y_pred = model_rf.predict(X_test)

#%% Criar arquivo de submissao

submission = pd.DataFrame(test['PassengerId'])
submission['Survived'] = y_pred

submission.to_csv('submission1.csv', index = False)


