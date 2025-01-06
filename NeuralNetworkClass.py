import numpy as np

from LayerClass import Layer
from RegularizationOptimizationClass import Regularization, Optimization
from DataProcessingClass import DataProcessing
from Functions import activation_functions, d_activation_functions, loss_functions, d_loss_functions

class NeuralNetwork:

    def __init__(self, layers_config, reg_config, opt_config):
        '''
        Class for the neural network  

        Args:
            layers_config (list): layers configuration as list:(dim_prev_layer, dim_layer, 
            activation_function, d_activation_function).
            reg_config (dict): regularization configuration.
            opt_config (dict): optimization configuration.
        '''
        self.regularizer = Regularization(**reg_config)
        self.layers, self.optimizers = self.initialize_layers(layers_config, opt_config)

    def initialize_layers(self, layers_config, opt_config):
        '''
        Function that iniliatizes all the layers in the neural network.

        Args:
            layers_config (list): layers configuration as list: 
                                  (dim_prev_layer, dim_layer, activation_function, d_activation_function). 
            opt_config (list): optimizations configuration as list: 
                               (opt_type, learning_rate_w, learning_rate_b, momentum, beta_1, beta_2, epsilon). 

        Returns:
            layers (list): list of the layers of the neural network.
            optimizers (list): list of instances of optimization class.
        '''
        layers = []
        optimizers = []
        for config in layers_config:
           layer = Layer(*config)
           optimizer = Optimization(layer.weights, layer.biases, regulizer = self.regularizer, **opt_config)
           layers.append(layer)
           optimizers.append(optimizer)
        return layers, optimizers


    def forward(self, input):
        '''
        Function that iterates the layer.forward_layer method through each layer in the list self.layers

        Args:
            input (array): in case of the input layer, it is the data batch. 
                           In case of a hidden layer, it is the output of the previous layer.

        Returns:
            input (array): the output of the current layer.
        '''
        for layer in self.layers:
            input = layer.forward_layer(input)
        return input

    def backward(self, loss_gradient):
        '''
        Function that iterates from the last layer to the first layer the layer.backward_layer method, 
        thus updating the weights and the biases for each layer.

        Args:
            loss_gradient (array): gradient loss with respect to the layer output.
        '''
        for layer, optimizer in reversed(list(zip(self.layers, self.optimizers))): 
            loss_gradient = layer.backward_layer(loss_gradient, optimizer)

    def reinitialize_weights(self):
        '''
        Function that re-initializes the weights layer-by-layer in case of need, e.g. with k-fold cross 
        validation after each cycle.
        '''
        for layer in self.layers:
            layer.initialize_weights_biases() # does it layer-by-layer


'''
np.random.seed(42)

# Configurazione dei layer: [(input_dim, output_dim, activation_function, d_activation_function), ...]
layers_config = [
    (5, 4, activation_functions['linear'], d_activation_functions['d_linear']),
    (4, 3, activation_functions['linear'], d_activation_functions['d_linear'])
]

# Configurazione della regolarizzazione
reg_config = {
    'Lambda_t': 0.01,
    'Lambda_l': 0.01,
    'alpha': 1e-4,
    'reg_type': 'elastic'
}

# Configurazione dell'ottimizzazione
opt_config = {
    'opt_type': 'NAG',
    'learning_rate_w': 0.001,
    'learning_rate_b': 0.001,
    'momentum': 0.9,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'epsilon': 1e-8,
}

nn = NeuralNetwork(layers_config, reg_config, opt_config)
x_tot = np.random.rand(10, 5)
target_tot = np.random.rand(10, 3)
#print(f"target tot: \n {target_tot}")
K = 3

data_split = DataProcessing(x_tot, target_tot, 0, K)
print(f'x_train \n {data_split.x_trains}')
epochs = 10
for i in range(epochs):
    loss = 0
    # for nei k fold
    for xx, target in zip(data_split.x_trains, data_split.target_trains):
        pred = nn.forward(xx)
        #print(f"target: \n {target}")
        #print(f"pred: \n {pred}")
        loss += loss_functions['mse'](target, pred)
        d_loss = d_loss_functions['d_mse'](target, pred)
        nn.backward(d_loss)
    loss /= K
    print(f"loss: \n {loss}")
'''