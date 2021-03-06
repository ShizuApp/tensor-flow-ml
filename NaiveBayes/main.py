import pandas as pd

# Read DataFrame
df = pd.read_table('SMSSpamCollection', sep='\t', names=['label', 'message'], header=None)

# Convert labels into binary variables
df['label'] = df.label.map({'ham':0, 'spam':1})

from sklearn.model_selection import train_test_split
X_train, y_train, X_test, y_test = \
    train_test_split(df['message'], df['label'], random_state=1)

from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer(stop_words='english')

training_data = count_vector.fit_transform(X_train)
testing_data = count_vector.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)

predictions = naive_bayes.predict(testing_data)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('Precision score: ', format(precision_score(y_test, predictions)))
print('Recall score: ', format(recall_score(y_test, predictions)))
print('F1 score: ', format(f1_score(y_test, predictions)))