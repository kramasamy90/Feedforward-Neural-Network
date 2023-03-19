import numpy as np
from ann import ann

def get_batch(X_train, y_train, batch_size):
    n_datapoints = y_train.shape[0]
    random_indices = np.random.choice(n_datapoints, batch_size, replace=False)
    X_train_batch = X_train[random_indices]
    y_train_batch = y_train[random_indices]
    return [X_train_batch, y_train_batch]

def compute_gradient(nn, X_train, y_train):
    X_train_batch, y_train_batch = get_batch(X_train, y_train, nn.batch_size)
    grad_Ws = []
    grad_bs = []
    for i in range(ann.num_layers + 1):
        grad_Ws.append(np.zeros([nn.Ws[i].shape[0], nn.Ws[i].shape[1]]))
        grad_bs.append(np.zeros([nn.bs[i].shape[0], nn.bs[i].shape[1]]))
    for i in range(nn.batch_size):
        nn.forward_prop(X_train_batch[i].flatten())
        nn.back_prop(y_train_batch[i])
        for i in range(ann.num_layers + 1):
            grad_Ws[i] += nn.grad_Ws[i]
            grad_bs[i] += nn.grad_bs[i]
    return grad_Ws, grad_bs

def batch_gd(nn, X_train, y_train):
    grad_Ws, grad_bs = compute_gradient(nn, X_train, y_train)    
    for i in range(ann.num_layers + 1):
        nn.Ws[i] -= ann.learning_rate * ann.weight_decay * nn.Ws[i]
        nn.Ws[i] -= ann.learning_rate * grad_Ws[i] / nn.batch_size
        nn.bs[i] -= ann.learning_rate * ann.weight_decay * nn.bs[i]
        nn.bs[i] -= ann.learning_rate * grad_bs[i] / nn.batch_size

def sgd(nn, X_train, y_train):
    nn.batch_size = 1
    batch_gd(nn, X_train, y_train)