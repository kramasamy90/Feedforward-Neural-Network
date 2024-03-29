import numpy as np
import matplotlib.pyplot as plt
# from ann import ann

#%%
# Activation functions and their derivatives.

def identity(x):
    return x

def d_identity(x):
    return 1

def softmax(x): # x is a numpy array.
    max_val = np.max(x)
    x_exp = np.exp(x - max_val)
    x_exp /= np.sum(x_exp)
    return x_exp

def d_softmax(x, l): # Derivative of softmax
    '''
    x: Numpy array.
    l: l is the category to which the input feature belongs to. This is the same notation used in lecture.
    '''

    y = softmax(x)
    _1 = np.zeros(y.shape)
    _1[l] = 1
    return (_1 * y[l] - y[l] * y)


def sigmoid(x):
    x = np.clip(x, -100, 100)
    return 1/(1+np.exp(-1 * x))

def d_sigmoid(x): # derivative of sigmoid
    y = sigmoid(x)
    return y * (1 - y)

def tanh(x):
    return np.tanh(x)

def d_tanh(x): # derivative of tanh
    y = np.tanh(x)
    return 1 - y ** 2

def relu(x):
    return np.maximum(0, x)

def d_relu(x): # derivative of relu
    return np.where(x > 0, 1, 0)

#%%
# Weight initialization.

def zero_init(n_rows, n_cols):
    return np.zeros([n_rows, n_cols])


def random_init(n_rows, n_cols, scale = 0.01):
    return scale * np.random.randn(n_rows, n_cols)

def xavier_init(n_rows, n_cols, scale = 1):
    return scale * np.random.randn(n_rows, n_cols) * np.sqrt(2 / (n_cols + n_rows))

#%%
# Functions to troubleshoot 

def get_norms_params(nn):
    norm = 0
    for W in nn.Ws:
        norm += np.linalg.norm(W) ** 2
    for b in nn.bs:
        norm += np.linalg.norm(b) ** 2
    return norm ** (0.5)

def get_norms_grads(nn):
    norm = 0
    for W in nn.grad_Ws:
        norm += np.linalg.norm(W) ** 2
    for b in nn.grad_bs:
        norm += np.linalg.norm(b) ** 2
    return norm ** (0.5)

#%% Plots

def view_weights_distribution(nn, num_layers, minval = -3, maxval = 3):
    fig, ax = plt.subplots(2, num_layers + 1)
    for i in range(num_layers + 1):
        ax[0][i].hist(nn.Ws[i].flatten(), range=(minval, maxval))
        ax[1][i].hist(nn.bs[i].flatten(), range=(minval, maxval))