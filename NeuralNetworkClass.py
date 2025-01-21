import numpy as np

from LayerClass import Layer
from RegularizationOptimizationClass import Regularization, Optimization

class NeuralNetwork:

    def __init__(self, layers_config, reg_config, opt_config, grid_search = False):
        '''
        Class for the neural network  

        Args:
            layers_config (list): layers configuration as a list of N layers, each defined by the following parameters:
                                  dim_prev_layer, dim_layer, activation_function, d_activation_function.
            reg_config (dict): a dictionary for regularization configuration defined by the following keys: Lambda, alpha, reg_type.
            opt_config (dict): a dictionary for optimization configuration defined by the following keys:
                               opt_type, learning_rate, momentum, beta_1, beta_2, epsilon.
        '''
        self.grid_search = grid_search
        self.regularizer = Regularization(**reg_config)
        if self.grid_search:
            self.layers, self.optimizers = self.initialize_layers_grid_search(layers_config, opt_config)
        else:
            self.layers, self.optimizers = self.initialize_layers_default(layers_config, opt_config)

        
    
    # per CUPUnitTest
    def initialize_layers_default(self, layers_config, opt_config):
      
        layers = []
        optimizers = []
        for config in layers_config:
           layer = Layer(*config)
           optimizer = Optimization(layer.weights, layer.biases, regulizer = self.regularizer, **opt_config)
           layers.append(layer)
           optimizers.append(optimizer)
        return layers, optimizers
    
    
    # per GridSearchClass
    
    def initialize_layers_grid_search(self, layers_config, opt_config):
        '''
        Function that iniliatizes all the layers in the neural network.
        Args:
            layers_config (list): layers configuration as a list of N layers, each defined by the following parameters:
                                  dim_prev_layer, dim_layer, activation_function, d_activation_function.
            opt_config (dict): a dictionary for optimization configuration defined by the following keys:
                               opt_type, learning_rate, momentum, beta_1, beta_2, epsilon.

        Returns:
            layers (list): list of the layers of the neural network.
            optimizers (list): list of optimization class instances, one for each layer of the neural network.
        '''
       
        layers = []
        optimizers = []
        for config in layers_config:
            # Decomposing GridSearchClass dictionary
            layer = Layer(
                dim_prev_layer=config['dim_prev_layer'],
                dim_layer=config['dim_layer'],
                activation_function=config['activation function'],
                d_activation_function=config['d_activation_function']
            )
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
            input (new array): the output of the current layer.
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
            optimizer.t += 1


    def reinitialize_net_and_optimizers(self):
        '''
        Function that re-initializes weights, biases and all the parameters of the optimizers layer-by-layer in case of need, 
        e.g. with k-fold cross validation after each cycle.
        '''
        for layer, optimizer in zip(self.layers, self.optimizers):
            layer.initialize_weights_biases()
            optimizer.initialization(layer.weights, layer.biases)
            optimizer.t = 1