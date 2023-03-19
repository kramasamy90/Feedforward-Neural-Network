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
        for j in range(ann.num_layers + 1):
            grad_Ws[j] += nn.grad_Ws[j]
            grad_Ws[j] += ann.learning_rate * ann.weight_decay * nn.Ws[j]
            grad_bs[j] += nn.grad_bs[j]
            grad_bs[j] += ann.learning_rate * ann.weight_decay * nn.bs[j]
    return grad_Ws, grad_bs

def batch_gd(nn, X_train, y_train, epochs):
    for i in range(epochs):
        grad_Ws, grad_bs = compute_gradient(nn, X_train, y_train)    
        for i in range(ann.num_layers + 1):
            nn.Ws[i] -= ann.learning_rate * grad_Ws[i] / nn.batch_size
            nn.bs[i] -= ann.learning_rate * grad_bs[i] / nn.batch_size

def sgd(nn, X_train, y_train, epochs):
    nn.batch_size = 1
    batch_gd(nn, X_train, y_train, epochs)

def momentum(nn, X_train, y_train, epochs):
    wu = []
    bu = []
    grad_Ws, grad_bs = compute_gradient(nn, X_train, y_train)    
    wu = grad_Ws
    bu = grad_bs
    for i in range(epochs):
        for j in range(len(nn.Ws)):
            nn.Ws[j] -= ann.learning_rate * wu[j]
            nn.bs[j] -= ann.learning_rate * bu[j]
        grad_Ws, grad_bs = compute_gradient(nn, X_train, y_train)    
        for j in range(len(wu)):
            wu[j] = nn.beta * wu[j] + grad_Ws[j]
            bu[j] = nn.beta * bu[j] + grad_bs[j]

def nesterov(nn, X_train, y_train, epochs):
    # Incomplete. Later.
    wu = []
    bu = []
    grad_Ws, grad_bs = compute_gradient(nn, X_train, y_train)    
    wu = grad_Ws
    bu = grad_bs
    for i in range(epochs):
        for j in range(len(nn.Ws)):
            nn.Ws[j] -= ann.learning_rate * wu[j]
            nn.bs[j] -= ann.learning_rate * bu[j]
        grad_Ws, grad_bs = compute_gradient(nn, X_train, y_train)    
        for j in range(len(wu)):
            wu[j] = nn.beta * wu[j] + grad_Ws[j]
            bu[j] = nn.beta * bu[j] + grad_bs[j]

def rmsprop(nn, X_train, y_train, epochs):
    wv = []
    bv = []
    grad_Ws, grad_bs = compute_gradient(nn, X_train, y_train)    
    wv = [(1 - ann.beta) * grad_Ws[i] ** 2 for i in range(len(grad_Ws))]
    bv = [(1 - ann.beta) * grad_bs[i] ** 2 for i in range(len(grad_bs))]
    for i in range(epochs):
        for j in range(len(nn.Ws)):
            nn.Ws[j] -= (ann.learning_rate / (np.sqrt(wv[j]) + ann.epsilon)) * grad_Ws[j]
            nn.bs[j] -= (ann.learning_rate / (np.sqrt(bv[j]) + ann.epsilon)) * grad_bs[j]
        grad_Ws, grad_bs = compute_gradient(nn, X_train, y_train)    
        wv = [ann.beta * wv[j] + (1 - ann.beta) * grad_Ws[j] ** 2 for j in range(len(grad_Ws))]
        bv = [ann.beta * bv[j] + (1 - ann.beta) * grad_bs[j] ** 2 for j in range(len(grad_bs))]

