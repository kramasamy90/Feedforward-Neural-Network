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

    def __init__(self, input_layer_width):

        self.input_layer_width = input_layer_width

        # Intialize
        self.top_layer_width = input_layer_width # Top layer need not be the output layer. 
        self.n_hidden_layers = 0
        self.contains_output_layer = False # Used later to ensure hidden layers are not added on top of output layer.
        self.Ws = []
        self.bs = []

        # Set default loss and activation functions
        self.activation_function = ann_utils.logistic
        self.d_activation_function = ann_utils.d_logistic
        self.output_activation_function = ann_utils.softmax
        self.loss = 'cross_entropy'

    def add_hidden_layer(self, width):
        if self.contains_output_layer:
            print("ERROR: Cannot add hidden layer on top of output layer!")
            return

        W = np.zeros(self.top_layer_width * width).reshape(width, self.top_layer_width)
        self.Ws.append(W)

        b = np.zeros(width).reshape(width, 1)
        self.bs.append(b)

        self.n_hidden_layers += 1
        self.top_layer_width = width
    
    def add_n_hidden_layers(self, n_hidden_layers, hidden_layer_width):
        '''
        Adds a block of hidden layers on top of the existing top layer, which could just be the input layer.
        The block has a depth of n_hidden_layers and width of hidden_layer_width
        '''
        self.hidden_layer_width = hidden_layer_width
        for i in range(n_hidden_layers):
            self.add_hidden_layer(hidden_layer_width)

    def add_output_layer(self, output_layer_width):
        self.add_hidden_layer(output_layer_width)
        self.n_hidden_layers -= 1 
        # Because the above line uses 'add_hidden_layer' function to add output layer.
        # And 'add_hidden_layer' function increases self.n_hidden_layers.
        self.contains_output_layer = True
    
    def forward_prop(self, X):
        x = X.flatten()
        self.x = x.reshape(len(x), 1)
        self.h = []
        self.a = []
        h = self.x
        for i in range(self.n_hidden_layers):
            a = self.Ws[i] @ h + self.bs[i]
            h = self.activation_function(a)
            self.a.append(a.reshape(len(a), 1))
            self.h.append(h.reshape(len(h), 1))
        al = self.Ws[self.n_hidden_layers] @ h + self.bs[self.n_hidden_layers]
        y = self.output_activation_function(al)
        self.al = al
        self.y = y
        return y

    def back_prop(self, y_actual):
        self.grad_Ws = []
        self.grad_bs = []
        if(self.loss == 'cross_entropy'):
            l = y_actual
            I_l = np.zeros(len(self.y)).reshape(len(self.y), 1)
            I_l[l][0] = 1
            grad_a = I_l - self.y
            grad_a = grad_a.reshape(len(grad_a), 1)
        
        L = self.n_hidden_layers # Total number of layers - 1, indexing starts from 0.
        for k in range(L, 0, -1):
            grad_W = grad_a @ self.h[k-1].T
            grad_b = grad_a
            grad_h = self.Ws[k].T @ grad_a
            grad_a = grad_h * self.d_activation_function(self.a[k-1])
            self.grad_Ws.append(grad_W)
            self.grad_bs.append(grad_b)
        
        grad_W = grad_a @ self.x.T
        self.grad_Ws.append(grad_W)
        self.grad_bs.append(grad_a) # grad_b = grad_a

        self.grad_Ws.reverse()
        self.grad_bs.reverse()
    
    def predict(self, X):
        return np.argmax(self.forward_prop(X))


if (__name__ == '__main__'):
    nn = ann(3)
    nn.add_n_hidden_layers(3, 2)
    nn.add_output_layer(2)
    x = np.array([1, 2, 3])
    y = np.array([0, 1])
    nn.forward_prop(x)
    nn.back_prop(y)