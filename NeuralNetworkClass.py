import numpy as np

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
        self.layers, self.optimizers = self.initialize_layers(layers_config, opt_config)

    def initialize_layers(self, layers_config, opt_config):
        '''
        Function that iniliatizes all the layers in the neural network.

        Args:
            layers_config (list): layers configuration as list: 
                                  (dim_prev_layer, dim_layer, activation_function, d_activation_function). 
            opt_config (list): optimizations configuration as list: 
                               (opt_type, learning_rate, momentum, beta_1, beta_2, epsilon). 

        Returns:
            layers (list): list of the layers of the neural network.
            optimizers (list): list of instances of optimization class.
        '''
        layers = []
        optimizers = []
        for config in layers_config:
           layer = Layer(*config)
           #print(f'layer w: {layer.weights}')
           optimizer = Optimization(layer.weights, layer.biases, regulizer = self.regularizer, **opt_config)
           #print(f'opt w: {optimizer.weights}')
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
            #print(f'pesi: \n {layer.weights}')
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