import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('appl.csv')
X = dataset.iloc[:, 0].values
y = dataset.iloc[:, 3].values

X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

from sklearn.svm import SVR
regressor = SVR()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
y_pred = sc_y.inverse_transform(y_pred)

y_test = sc_y.inverse_transform(y_test) 
X_test = sc_X.inverse_transform(X_test)
