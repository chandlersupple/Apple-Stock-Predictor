# Chandler Supple, 6/9/2018
# Dates were converted to integers in Google Sheets with the 'Number' tool.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('apple_stock_dataset.csv')
X = dataset.iloc[:, 0].values
y = dataset.iloc[:, 3].values

X_future = [[43258], # Additional dates to be predicted.
             [43259],
             [43260],
             [43261],
             [43263],
             [43265],
             [43266],
             [43267],
             [43268],
             [43269],
             [43270],
             [43271],
             [43272],
             [43273],
             [43274],
             [43275],
             [43276],
             [43271],
             [43291],
             [43313],
             [43374],
             [43435],
             [43466],
             [43497]]

X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

from sklearn.preprocessing import StandardScaler # Scaling data.
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
X_future = sc_X.transform(X_future)
y = sc_y.fit_transform(y)

from sklearn.svm import SVR # Creating and fitting regressor.
regressor = SVR()
regressor.fit(X, y)

y_pred = regressor.predict(X) # Making predictions for the actual data.
y_pred = sc_y.inverse_transform(y_pred)
X_future_pred = regressor.predict(X_future)
X_future_pred = sc_y.inverse_transform(X_future_pred)

X_grid = np.arange(min(X), max(X), 0.01) # Graphing the resulting predictions against the actual data.
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.show()
