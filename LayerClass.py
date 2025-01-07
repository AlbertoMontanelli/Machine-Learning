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
        self.initialize_weights_biases()


    def initialize_weights_biases(self):
        '''
        Function that initializes the Weights and the Biases of the network
        '''
        self.weights = np.random.uniform(low=-1/np.sqrt(self.dim_prev_layer), high=1/np.sqrt(self.dim_prev_layer), 
                                         size=(self.dim_prev_layer, self.dim_layer))
        self.biases = np.zeros((1, self.dim_layer))


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
    
    
    def backward_layer(self, loss_gradient, optimizer):
        '''
        Function that computes the gradient loss and updates the weights and biases by the learning rule for the single layer

        Args:
            loss_gradient (array): gradient loss with respect to the layer output.
            regularizer (Regularization): istance of Regularization class.
            optimizer (Optimization): istance of Optimization class.

        Returns:
            sum_delta_weights (array): loss_gradient for hidden layer
        '''
        #print("Pesi Layer prima:", self.weights)

        sum_delta_weights = optimizer.optimization(self.input, loss_gradient, self.d_activation_function)
        #print("Pesi layer dopo:", self.weights)
        return sum_delta_weights