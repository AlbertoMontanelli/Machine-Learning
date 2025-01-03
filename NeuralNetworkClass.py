import numpy as np

from Functions import activation_functions, d_activation_functions
from LayerClass import Layer
from RegularizationOptimizationClass import Regularization, Optimization

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
        self.optimizer = Optimization(regulizer = self.regularizer, **opt_config)
        self.layers = self.initialize_layers(layers_config)

    def initialize_layers(self, layers_config):
        '''
        Function that iniliatizes all the layers in the neural network.

        Args:
            layers_config (list): layers configuration as list:(dim_prev_layer, dim_layer, 
            activation_function, d_activation_function).  

        Returns:
            layers (list): list of the layers of the neural network.
        '''
        layers = []
        for config in layers_config:
           layer = Layer(*config)
           self.optimizer.initialization(layer.weights, layer.biases)
           layers.append(layer)

    def data_split(self, x_tot, target, test_split):
        '''
        Function that splits the input data into two sets: training & validation set, test set.
        
        Args:
            x_tot: total data given as input.
            target: total data labels given as input.
            test_split: percentile of test set with respect to the total data.
        
        Returns:
            x_train_val: training and validation set extracted from input data.
            target_train_val: training and validation set labels.
            x_test_val: test set extracted from input data.
            target_test_val: test set for input data labels.
        '''

        num_samples = x_tot.shape[0] # the total number of the examples in input = the number of rows in the x_tot matrix
        test_size = int(num_samples * test_split) # the number of the examples in the test set

        x_test = x_tot[:test_size]
        target_test = target[:test_size]
        x_train_val = x_tot[test_size:]
        target_train_val = target[test_size:]

        return x_train_val, target_train_val, x_test, target_test

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
        for layer in reversed(self.layers):
            loss_gradient = layer.backward_layer(loss_gradient, self.optimizer)

    def reinitialize_weights(self):
        '''
        Function that re-initializes the weights layer-by-layer in case of need, e.g. with k-fold cross 
        validation after each cycle.
        '''
        for layer in self.layers:
            layer.initialize_weights_biases() # does it layer-by-layer


## Unit test for NeuralNetworkClass

# Configurazione dei layer: [(input_dim, output_dim, activation_function, d_activation_function), ...]
layers_config = [
    (10, 20, activation_functions['sigmoid'], d_activation_functions['d_sigmoid']),
    (20, 10, activation_functions['tanh'], d_activation_functions['d_tanh']),
    (10, 5, activation_functions['ReLU'], d_activation_functions['d_ReLU'])
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
    'learning_rate_w': 0.001,
    'learning_rate_b': 0.001,
    'momentum': 0.9,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'epsilon': 1e-8,
    'opt_type': 'NAG'
}

# Inizializza la rete neurale
network = NeuralNetwork(layers_config, reg_config, opt_config)


