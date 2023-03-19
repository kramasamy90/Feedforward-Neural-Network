import ann_utils
import gd

arg_short_to_long = {
    'wp': 'wandb_project', # str
    'we': 'wandb_entity', # str
    'd': 'dataset', # 'mnist', 'fashion_mnist', 
    'e': 'epochs', # int
    'b': 'batch_size', # int
    'l': 'loss', # ''mean_squared_error', 'cross_entropy'. 
    'o': 'optimizer', # 'sgd, 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'.
    'lr': 'learning_rate', # float
    'm': 'momentum', # float
    'beta': 'beta', # float
    'beta1': 'beta1', # float
    'beta2': 'beta2', # float
    'eps': 'epsilon', # float
    'w_d': 'weight_decay', # float
    'w_i': 'weight_init', # 'random', 'Xavier'
    'nhl': 'num_layers', # int
    'sz': 'hidden_size', # int
    'a': 'activation' # identity,sigmoid, tanh, ReLU
}

default_args = {
    'wandb_project': 'myprojectname',
    'wandb_entity': 'myname',
    'dataset': 'fashion_mnist',
    'epochs': '1',
    'batch_size': '4',
    'loss': 'cross_entropy',
    'optimizer': 'sgd',
    'learning_rate': '0.1',
    'momentum': '0.5',
    'beta': '0.5',
    'beta1': '0.5',
    'beta2': '0.5',
    'epsilon': '0.000001',
    'weight_decay': '0.0',
    'weight_init': 'random',
    'num_layers': '1',
    'hidden_size': '4',
    'activation': 'sigmoid'
}

activation= {
    'identity': ann_utils.identity,
    'sigmoid': ann_utils.sigmoid,
    'tanh': ann_utils.tanh,
    'ReLU': ann_utils.relu
}

d_activation= {
    'identity': ann_utils.d_identity,
    'sigmoid': ann_utils.d_sigmoid,
    'tanh': ann_utils.d_tanh,
    'ReLU': ann_utils.d_relu
}

weight_init= {
    'random': ann_utils.random_init,
    'Xavier': ann_utils.xavier_init
}

optimizer = {
    'sgd': gd.sgd
}