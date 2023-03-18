import numpy as np
import ann

def get_batch(X_train, y_train, batch_size):
    n_datapoints = y_train.shape[0]
    random_indices = np.random.choice(n_datapoints, batch_size, replace=False)
    X_train_batch = X_train[random_indices]
    y_train_batch = y_train[random_indices]
    return [X_train_batch, y_train_batch]

def batch_gd(nn, X_train, y_train):
    L = nn.n_hidden_layers + 1
    X_train_batch, y_train_batch = get_batch(X_train, y_train, nn.batch_size)
    grad_Ws = []
    grad_bs = []
    for i in range(L):
        grad_Ws.append(np.zeros([nn.Ws[i].shape[0], nn.Ws[i].shape[1]]))
        grad_bs.append(np.zeros([nn.bs[i].shape[0], nn.bs[i].shape[1]]))
    for i in range(nn.batch_size):
        nn.forward_prop(X_train_batch[i].flatten())
        nn.back_prop(y_train_batch[i])
        for i in range(L):
            grad_Ws[i] += nn.grad_Ws[i]
            grad_bs[i] += nn.grad_bs[i]
    
    for i in range(L):
        nn.Ws[i] -= nn.learning_rate * grad_Ws[i] / nn.batch_size
        nn.bs[i] -= nn.learning_rate * grad_bs[i] / nn.batch_size

def sgd(nn, eta, X_train, y_train):
    nn.batch_size = 1
    batch_gd(nn, X_train, y_train)