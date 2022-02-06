import pandas as pd
from sklearn import linear_model

# Assign the data to predictor and outcome variables
train_data = pd.read_csv('data.csv', header=None)
X = train_data.iloc[:,:-1]
y = train_data.iloc[:,-1]

# Assign 90% of data as training data
pos = int(len(y) * 0.9)
training_features = X[:pos]
training_labels = y[:pos]
# Other 10% as testing data
testing_features = X[pos:]
testing_labels = y[pos:]

# Create the linear regression model with lasso regularization
lasso_reg = linear_model.Lasso()
lasso_reg.fit(training_features, training_labels)

clf = linear_model.LinearRegression()
clf.fit(training_features, training_labels)

# Retrieve and print out the coefficients from the regression model
reg_coef = lasso_reg.coef_
lin_coef = clf.coef_

print("Expected:\n", testing_labels.values)
print("Linear:\n", clf.predict(testing_features))
print("Lasso:\n", lasso_reg.predict(testing_features))