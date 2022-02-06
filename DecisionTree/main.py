from statistics import quantiles
from decisiontree import DTree
import pandas as pd

db = pd.read_csv('ml-bugs.csv')

# List all categories in the data frame (headers)
ctgz = list(db.columns)

dt = DTree()

# Find entropy for each category
for ctg in ctgz:
    #quantts = dt.counter(db[ctg].values).values()
    print(dt.counter(db[ctg]))
    #print(ctg, ' -> ', dt.entropy(quantts))