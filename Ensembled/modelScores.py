import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Read in dataset
df = pd.read_table('SMSSpamCollection.csv',
                   sep='\t', 
                   header=None, 
                   names=['label', 'sms_message'])

# Fix response value
df['label'] = df.label.map({'ham':0, 'spam':1})

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(df['sms_message'], 
                                                    df['label'], 
                                                    random_state=1)

# Transform training and testing data to matrix
count_vector = CountVectorizer()
training_data = count_vector.fit_transform(X_train)
testing_data = count_vector.transform(X_test)

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB

bagg = BaggingClassifier(n_estimators=200)
ranf= RandomForestClassifier(n_estimators=200)
adab = AdaBoostClassifier(n_estimators=300, learning_rate=0.2)
naiv = MultinomialNB()

bagg.fit(training_data, y_train)
ranf.fit(training_data, y_train)
adab.fit(training_data, y_train)
naiv.fit(training_data, y_train)

bagp = bagg.predict(testing_data)
ranp = ranf.predict(testing_data)
adap = adab.predict(testing_data)
naip = naiv.predict(testing_data)

def print_metrics(y_true, preds, model_name=None):
    '''
    y_true - the y values that are actually true in the dataset (NumPy array or pandas series)
    preds - the predictions for those values from some model (NumPy array or pandas series)
    model_name - (str - optional) a name associated with the model if you would like to add it to the print statements 
    
    OUTPUT:
    None - prints the accuracy, precision, recall, and F1 score
    '''
    print(model_name)
    print('Accuracy score: ', format(accuracy_score(y_true, preds)))
    print('Precision score: ', format(precision_score(y_true, preds)))
    print('Recall score: ', format(recall_score(y_true, preds)))
    print('F1 score: ', format(f1_score(y_true, preds)))
    print('-------------------------------\n')

print_metrics(y_test, bagp, 'Bagging')
print_metrics(y_test, ranp, 'RandomForest')
print_metrics(y_test, adap, 'AdaBoost')
print_metrics(y_test, naip, 'NaiveBayes')