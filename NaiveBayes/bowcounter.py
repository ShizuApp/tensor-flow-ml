documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']

lower_case_documents = []
for i in documents:
    lower_case_documents.append(i.lower())

sans_punctuation_documents = []

for i in lower_case_documents:
    i = i.replace(',', '')
    i = i.replace('?', '')
    i = i.replace('!', '')
    i = i.replace('.', '')

    sans_punctuation_documents.append(i)
    

preprocessed_documents = []
for i in sans_punctuation_documents:
    preprocessed_documents.append(i.split())


frequency_list = []
from collections import Counter

for i in preprocessed_documents:
    frequency_list.append(Counter(i))
    
print(frequency_list)


"""
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer(stop_words='english')

doc_array = count_vector.fit_transform(documents).toarray()
feature_names = count_vector.get_feature_names()

frequency_matrix = pd.DataFrame(doc_array, columns=feature_names)

print(frequency_matrix)
"""