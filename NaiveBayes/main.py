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