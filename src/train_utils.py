import numpy as np
from ann import ann

def get_classification_accuracy(nn, X_valid, y_valid):
    n = len(y_valid.flatten())
    correct = 0.0
    for i in range(n):
        if nn.predict(X_valid[i]) == y_valid[i]:
            correct += 1
    return correct / n