import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

# Assign the data to predictor and outcome variables
train_data = pd.read_csv('data.csv', header=None)
X = train_data.iloc[:,:-1]
y = train_data.iloc[:,-1]

# Create the standardization scaling object
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso_reg = linear_model.Lasso()
lasso_reg.fit(X_scaled, y)

# Retrieve and print out the coefficients from the regression model
reg_coef = lasso_reg.coef_
print(reg_coef)