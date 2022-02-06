import pandas as pd
import numpy as np


def stepFunction(n):
    if n >= 0:
        return 1
    return 0                                                                                                                              


def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b))


def perceptronStep(X, y, W, b, learn_rate = 0.01):
    for i in range(len(X)):
        y_hat = prediction(X[i], W, b)

        if y[i]-y_hat == 1:
            b += learn_rate
            for j in range(len(W)):
                W[j] += X[i][j] * learn_rate              
        elif y[i]-y_hat == -1:
            b -= learn_rate
            for j in range(len(W)):
                W[j] -= X[i][j] * learn_rate

    return W, b
    
    

def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25):
    xmin, xmax = min(X[0]), max(X[0])

    # Generate weights and the bias
    np.random.seed(42)
    W = np.random.rand(len(X[0]))
    b = np.random.rand(1)[0] + xmax

    for _ in range(num_epochs):
        W, b = perceptronStep(X, y, W, b, learn_rate)
         
    print(W, b)



def main():
    # Read the database
    db = pd.read_csv('data.csv', header=None)
    X = db.iloc[:,:-1].values # features
    y = db.iloc[:,-1].values # labes
    
    trainPerceptronAlgorithm(X, y)



if __name__ == '__main__':
    main()