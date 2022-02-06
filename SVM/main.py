from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Read the data
data = np.asarray(pd.read_csv('data.csv', header=None))

# Split data
from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = \
    train_test_split(data[:,0:2], data[:,2])

# Create and fit the model
model = SVC(gamma=27)
model.fit(train_features, train_labels)

# Make predictions
y_pred = model.predict(test_features)

# Calculate the accuracy and assign it to the variable acc.
acc = accuracy_score(test_labels, y_pred)

print(acc)