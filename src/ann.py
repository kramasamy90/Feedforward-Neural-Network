import numpy as np
import ann_utils

class ann:

    '''
    This class ann starts with input layer and is built layer-by-layer.
    * 'add_hidden_layer' adds one layer on top of the existing one.
    * 'add_n_hidden_layer' adds n layers by calling 'add_hidden_layer' function.
    * 'add_output_layer' add the output layer using 'add_hidden_layer' function.
    * 'add_output_layer' does some book-keeping after calling 'add_hidden_layer' function,
      to ensure the self.n_hidden_layers does not increase after adding the output layer.
    * 'forward_prop' and 'back_prop' does what their name indicates.
    '''
    # Input parameters.
    batch_size = 4
    learning_rate = 0.1
    momentum = 0.5
    beta = 0.5
    beta1 = 0.5
    beta2 = 0.5
    weight_decay = 0.0
    epsilon = 0.000001
    loss = 'cross_entropy'

    # Functions
    weight_init = ann_utils.random_init
    activation = ann_utils.sigmoid
    d_activation = ann_utils.d_sigmoid
    output_activation = ann_utils.softmax
    d_output_activation = ann_utils.d_softmax

    # ANN structure
    num_layers = 5
    hidden_size = 4

    def __init__(self, input_layer_width, output_layer_width):
        self.input_layer_width = input_layer_width
        self.output_layer_width = output_layer_width
        self.top_layer_width = self.input_layer_width
        self.Ws = []
        self.bs = []
        self.add_n_hidden_layers()


    def add_hidden_layer(self, width):
        W = ann.weight_init(width, self.top_layer_width)
        self.Ws.append(W)

        b = ann.weight_init(width, 1)
        self.bs.append(b)

        self.top_layer_width = width
    
    def add_output_layer(self):
        self.add_hidden_layer(self.output_layer_width)
        # Because the above line uses 'add_hidden_layer' function to add output layer.
        # And 'add_hidden_layer' function increases self.n_hidden_layers.
        self.contains_output_layer = True

    def add_n_hidden_layers(self):
        for i in range(self.num_layers):
            self.add_hidden_layer(ann.hidden_size)
        self.add_output_layer()
    
    def forward_prop(self, X):
        x = X.flatten()
        self.x = x.reshape(len(x), 1)
        self.h = []
        self.a = []
        h = self.x
        for i in range(ann.num_layers):
            a = self.Ws[i] @ h + self.bs[i]
            h = ann.activation(a)
            self.a.append(a.reshape(len(a), 1))
            self.h.append(h.reshape(len(h), 1))
        al = self.Ws[ann.num_layers] @ h + self.bs[ann.num_layers]
        y = ann.output_activation(al)
        self.al = al
        self.y = y
        return y

    def back_prop(self, y_actual):
        self.grad_Ws = []
        self.grad_bs = []
        print(y_actual)
        if(ann.loss == 'cross_entropy'):
            l = y_actual
            I_l = np.zeros(self.y.shape)
            I_l[l][0] = 1
            grad_a = self.y - I_l
            grad_a = grad_a.reshape(len(grad_a), 1)
        
        if(ann.loss == 'mean_squared_error'):
            grad_y =  self.y - y_actual
            dy_da = ann.d_output_activation(self.al)
            grad_a = grad_y * dy_da
        
        L = ann.num_layers # Total number of layers - 1, indexing starts from 0.
        for k in range(L, 0, -1):
            grad_W = grad_a @ self.h[k-1].T
            grad_b = grad_a
            grad_h = self.Ws[k].T @ grad_a
            grad_a = grad_h * ann.d_activation(self.a[k-1])
            self.grad_Ws.append(grad_W)
            self.grad_bs.append(grad_b)
        
        grad_W = grad_a @ self.x.T
        self.grad_Ws.append(grad_W)
        self.grad_bs.append(grad_a) # grad_b = grad_a

        self.grad_Ws.reverse()
        self.grad_bs.reverse()
    
    def predict(self, X):
        return np.argmax(self.forward_prop(X))
