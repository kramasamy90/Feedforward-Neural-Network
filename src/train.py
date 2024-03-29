import sys
import numpy as np
from ann import ann
import train_utils
import maps

args = maps.default_args

# Read and store argument in a map/dict.
for i in range(1, int((len(sys.argv) - 1)/2) + 1):
    arg = sys.argv[2 * i - 1]
    val = sys.argv[2 * i]
    if(arg[1] == '-'):
        arg = arg[2:]
        args[arg] = val
    else:
        arg = maps.arg_short_to_long[arg[1:]]
    args[arg] = val

if args['dataset'] == 'fashion_mnist':
    from keras.datasets import fashion_mnist
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

for key in args:
    print(key, ' = ', args[key])

# Normalize data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Meta information about the data.
n_train = len(y_train.flatten())
n_test = len(y_test.flatten())
input_dim = len(X_train[0].flatten())
output_dim = 10

# Split train data to train and validation sets.
ind = [i for i in range(n_train)]
np.random.shuffle(ind)
m = int(n_train * 0.1)
X_valid = X_train[ind[1: m]]
y_valid = y_train[ind[1: m]]
X_train = X_train[ind[m:]]
y_train = y_train[ind[m:]]

import wandb
wandb.login()
project_name = args['wandb_project']
entity = args['wandb_entity']

epochs = int(args['epochs'])
optimizer = maps.optimizer[args['optimizer']]

ann.batch_size = int(args['batch_size'])
ann.learning_rate = float(args['learning_rate'])
ann.momentum = float(args['momentum'])
ann.beta = float(args['beta'])
ann.beta1 = float(args['beta1'])
ann.beta2 = float(args['beta2'])
ann.epsilon = float(args['epsilon'])
ann.weight_decay = float(args['weight_decay'])
ann.hidden_size = int(args['hidden_size'])

ann.weight_init = maps.weight_init[args['weight_init']]
ann.activation = maps.activation[args['activation']]
ann.d_activation = maps.d_activation[args['activation']]
ann.loss = args['loss']

def train(config):

    nn = ann(input_dim, output_dim)

    optimizer(nn, X_train, y_train, epochs)

    performance = []
    performance.append(train_utils.get_classification_accuracy(nn, X_test, y_test))
    performance.append(train_utils.get_classification_accuracy(nn, X_valid, y_valid))
    performance.append(train_utils.get_loss(nn, X_test, y_test))
    performance.append(train_utils.get_loss(nn, X_valid, y_valid))

    return performance

def main():
    wandb.init(project = project_name, entity= entity)
    performance = train(wandb.config)
    wandb.log({'acc': performance[0], 'val_acc': performance[1], 'loss': performance[2], 'val_loss': performance[3]})

sweep_configuration = {
    'method': 'bayes',
    'metric': {
        'goal': 'minimize',
        'name': 'val_loss'
    },
    'parameters': {
        'placeholder': {'values': [0]},
    }
}

sweep_id = wandb.sweep(sweep = sweep_configuration, project = project_name)
wandb.agent(sweep_id, function = main, count = 1)