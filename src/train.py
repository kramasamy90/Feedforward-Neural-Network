import sys
from ann import ann
import maps

args = maps.default_args

# Read and store argument in a map/dict.

for i in range(1, int((len(sys.argv) - 1)/2) + 1):
    print(i)
    arg = sys.argv[2 * i - 1]
    print(arg)
    val = sys.argv[2 * i]
    if(arg[1] == '-'):
        arg = arg[2:]
        args[arg] = val
    else:
        arg = maps.arg_short_to_long[arg[1:]]
    args[arg] = val

epochs = args['epochs']
optimizer = maps.optimizer_map[args['optimizer']]

ann.batch_size = int(args['batch_size'])
ann.learning_rate = float(args['learning_rate'])
ann.momentum = float(args['momentum'])
ann.beta = float(args['beta'])
ann.beta1 = float(args['beta1'])
ann.beta2 = float(args['beta2'])
ann.epsilon = float(args['epsilon'])
ann.weight_decay = float(args['weight_decay'])

ann.weight_init = maps.weight_init[args['weight_init']]
ann.activation = maps.activation[args['activation']]
ann.d_activation = maps.d_activation[args['activation']]

if args['dataset'] == 'fashion_mnist':
    from keras.datasets import fashion_mnist
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    input_dim = len(X_train.flatten())
    output_dim = 10

nn = ann(input_dim, output_dim)

for i in range(epochs):
    optimizer(nn)


