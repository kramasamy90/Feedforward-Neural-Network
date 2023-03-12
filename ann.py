import numpy as np
import functions

class ann:
    '''
    This class ann starts with input layer and is built layer-by-layer.
    '''
    def __init__(self, input_layer_width):
        self.input_layer_width = input_layer_width
        self.top_layer_width = input_layer_width # Top layer need not be the output layer. 
        self.n_hidden_layers = 0
        self.contains_output_layer = False # Used later to ensure hidden layers are not added on top of output layer.
        self.Ws = []
        self.bs = []

    def add_hidden_layer(self, width):
        if self.contains_output_layer:
            print("ERROR: Cannot add hidden layer on top of output layer!")
            return

        W = np.zeros(self.top_layer_width * width).reshape(self.top_layer_width, width)
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
        self.n_hidden_layers -= 1 # Because the above line uses 'add_hidden_layer' function to add output layer.
        self.contains_output_layer = True
        self.top_layer_width = output_layer_width
    
    def f_prop(self, x):
        h = x
        for i in range(self.n_hidden_layers):
            a = self.Ws[i] @ h + self.bs[i]
            # h = self.(a)
        al = self.Ws[self.n_hidden_layers] @ h + self.bs[self.n_hidden_layers]
        y = functions.softmax(al)
        return y



if (__name__ == '__main__'):
    nn = ann(3)
    nn.add_n_hidden_layers(3, 2)
    nn.add_output_layer(2)
    print(nn.Ws)
    print(nn.n_hidden_layers)
    print(len(nn.Ws))