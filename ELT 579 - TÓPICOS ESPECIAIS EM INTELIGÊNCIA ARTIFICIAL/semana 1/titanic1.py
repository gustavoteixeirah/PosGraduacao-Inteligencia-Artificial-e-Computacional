#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 21:09:47 2023

@author: teixeira
"""


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
    return X


X_train = criar_features(X_train)
X_test = criar_features(X_test)



#%% Selecionar as features


features = ['Pclass', 'Age', 'SibSp',
       'Parch', 'Fare', 'mulher', 'porto', 'ehCrianca', 'ehIdoso']
X_train = X_train[features]
X_test = X_test[features]

y_train = train['Survived']

#%% Visualização

for i in X_train.columns:
    plt.hist(X_train[i])
    plt.title(i)
    plt.show()
  
#%% Groupy

gp = train.groupby(['Survived']).count()

#%% pivot_table

table = pd.pivot_table(train, index = ['Survived'], columns = ['Pclass'], values = 'PassengerId', aggfunc = 'count')





#%% Padronizacao das variaveis
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


#%% Modelo e validacao cruzada

# Logistic Regression
from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix
# from sklearn.preprocessing import StandardScaler
# pip install scikit-learn
model_lr = LogisticRegression(max_iter = 1000, random_state=0)
score = cross_val_score(model_lr, X_train, y_train, cv = 10)

model_lr_score =  np.mean(score)
print("LogisticRegression: ", model_lr_score)

#%% Naive Bayes

from sklearn.naive_bayes import GaussianNB

model_nb = GaussianNB()
score = cross_val_score(model_nb, X_train, y_train, cv = 10)

model_nb_score = np.mean(score)
print("GaussianNB: ", model_nb_score)

#%% Nearest Neighbors Classification

from sklearn.neighbors import KNeighborsClassifier

model_knc = KNeighborsClassifier(n_neighbors=10)
score = cross_val_score(model_knc, X_train, y_train, cv = 10)

model_knc_score = np.mean(score)
print("Nearest Neighbors Classification: ", model_knc_score)

#%% SVC

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

model_svc = SVC(C=1, kernel='linear', degree=2, gamma=0.01)
score = cross_val_score(model_svc, X_train, y_train, cv = 10)

model_svc_score = np.mean(score)
print("SVC: ", model_nb_score)


#%% Decision tree

from sklearn.tree import DecisionTreeClassifier

model_dt = DecisionTreeClassifier(random_state=0, max_depth=3)
score = cross_val_score(model_dt, X_train, y_train, cv = 10)

model_dt_score = np.mean(score)
print("Decision tree: ", model_dt_score)

#%% Random Forest
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(n_estimators=80, max_depth=3, random_state=0)

score = cross_val_score(model_rf, X_train, y_train, cv = 10)

model_rf_score = np.mean(score)
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
                                      random_state=0)

    score = cross_val_score(model_rf, X_train, y_train, cv = 10)

    mean_score = np.mean(score)
    
    return -mean_score
otimos = gp_minimize(treinar_modelo, parametros, random_state=0, verbose=1, n_calls=50, n_random_starts=10)


#%% Ensamble

from sklearn.ensemble import VotingClassifier

model_voting = VotingClassifier(estimators==[('LR', model_lr),
                                             ('KNN', model_knc),
                                             ('SVC', model_svc),
                                             ('RF', model_rf)],
                                voting='hard')

model_voting.fit(X_train, y_train)

 score = cross_val_score(model_voting, X_train, y_train, cv = 10)

 model_voting_score = np.mean(score)
 print("Decision tree: ", model_voting_score)


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











