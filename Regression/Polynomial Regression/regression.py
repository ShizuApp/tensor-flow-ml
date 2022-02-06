import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd


db = pd.read_csv('data.csv')
X = db[['Var_X']].values
y = db[['Var_Y']].values

poly = PolynomialFeatures(4)
newX = poly.fit_transform(X)

model = LinearRegression()
model.fit(newX, y)


fg = plt.figure()
vals = np.linspace(min(X), max(X), 100)
newVlas = poly.fit_transform(vals)
plt.plot(vals, model.predict(newVlas))

plt.scatter(X, y)


plt.show()



