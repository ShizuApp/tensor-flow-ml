import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Assign the data to predictor and outcome variables
train_data = pd.read_csv('data.csv')
X = train_data[['Var_X']].values
y = train_data[['Var_Y']].values

poly_feat = PolynomialFeatures(4)
X_poly = poly_feat.fit_transform(X)

# Create polynomial features
# Create a PolynomialFeatures object, then fit and transform the predictor feature
poly_feat = PolynomialFeatures(4)
X_poly = poly_feat.fit_transform(X)

# Make and fit the polynomial regression model
# Create a LinearRegression object and fit it to the polynomial predictor features
poly_model = LinearRegression(fit_intercept=False)
poly_model.fit(X_poly, y)

plt.scatter(X, y)
rawVal = np.linspace(min(X), max(X), 100)
values = poly_feat.fit_transform(rawVal)
predictions = poly_model.predict(values)
plt.plot(rawVal, predictions)

plt.show()