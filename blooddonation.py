# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 16:30:43 2018

@author: Lenovo
"""

import numpy as np
import pandas as pd

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

#deleting first column
X = df_train.iloc[:, 1:5].values
y = df_train.iloc[:, 5].values

#x for prediction
X_pred = df_test.iloc[:, 1:].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
X_pred = sc_X.transform(X_pred)

from sklearn.ensemble import GradientBoostingClassifier
gdc = GradientBoostingClassifier()
gdc.fit(X_train, Y_train)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, Y_train)

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, gdc.predict(X_test)))
print(accuracy_score(Y_test, rfc.predict(X_test)))

from sklearn.metrics import log_loss
print(log_loss(Y_test, gdc.predict(X_test)))
print(log_loss(Y_test, rfc.predict(X_test)))

submission = gdc.predict(X_pred)