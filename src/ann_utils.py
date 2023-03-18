import numpy as np

#%%
# Activation functions and their derivatives.

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

    _1 = np.zeros(x.shape[0])
    _1[l] = 1
    y = softmax(x)
    return (_1 * y[l] - y[l] * y)


def logistic(x):
    x[x < -100] = -100
    return 1/(1+np.exp(-1 * x))
    pass

def d_logistic(x): # derivative of sigmoid
    y = logistic(x)
    return y * (1 - y)

def tanh(x):
    return np.tanh(x)

def d_tanh(x): # derivative of tanh
    y = np.tanh(x)
    return 1 - y ** 2

def basic_relu(x): 
    return max(x, 0.0)

v_relu = np.vectorize(basic_relu)

def relu(x):
    return v_relu(x)

def basic_d_relu(x):
    if x <= 0: 
        return 0
    else:
        return 1

v_d_relu = np.vectorize(basic_d_relu)

def d_relu(x): # derivative of relu
    return v_d_relu(x)

#%%
# Weight initialization.

def zero_init(n_rows, n_cols):
    return np.zeros([n_rows, n_cols])


def random_init(n_rows, n_cols, scale = 0.01):
    return scale * np.random.randn(n_rows, n_cols)

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