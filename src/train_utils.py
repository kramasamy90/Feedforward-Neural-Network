import numpy as np
import math
from ann import ann

def get_classification_accuracy(nn, X_valid, y_valid):
    n = len(y_valid.flatten())
    correct = 0.0
    for i in range(n):
        if nn.predict(X_valid[i]) == y_valid[i]:
            correct += 1
    return correct / n

def get_loss(nn, X, y):
    n = len(y.flatten())
    loss = 0.0
    for i in range(n):
        l = y[i]
        nn.forward_prop(X[i])
        loss += -math.log(nn.y[l])
    
    return loss / n