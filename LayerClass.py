import numpy as np

class Layer:

    def __init__(
            self, 
            dim_prev_layer, 
            dim_layer, 
            activation_function, 
            d_activation_function
            ):
        '''
        Class for creation of the layers

        Args:
            dim_prev_layer (int): in case of the input layer, number of features. In case of a hidden layer, number of units of the 
                                  previous layer.
            dim_layer (int): number of units of the current layer.
            activation_function (func): activation function used in the layer.
            d_activation_function (func): derivative of the activation function used in the layer.

        '''
        self.dim_prev_layer = dim_prev_layer
        self.dim_layer = dim_layer
        self.activation_function = activation_function
        self.d_activation_function = d_activation_function
        self.initialize_parameters()

    def initialize_parameters(self):
        '''
        Function that initializes the Weights and the Biases of the network
        '''
        self.weights = np.random.uniform(low=-1/np.sqrt(self.dim_layer), high=1/np.sqrt(self.dim_prev_layer), 
                                         size=(self.dim_prev_layer, self.dim_layer))
        self.biases = np.zeros((1, self.output_size))

    def forward_layer(self, input):
        '''
        Function that computes the output of the current layer

        Args:
            input (array): in case of the input layer, it is the data batch. 
                           In case of a hidden layer, it is the output of the previous layer.
        
        Returns:
            output (array): computation of the output of the current layer.
        '''
        self.input = input
        self.net = np.dot(self.input, self.weights) + self.biases
        output = self.activation_function(self.net)
        return output