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


    def xavier_normal(self, shape, n_in, n_out):
        '''
        scrivere doc
        '''
        stddev = np.sqrt(2 / (n_in + n_out))
        return np.random.normal(0, stddev, shape)
    

    def initialize_weights_biases(self):
        '''
        Function that initializes the Weights and the Biases of the network
        '''
        self.weights = np.random.uniform(low=-1/np.sqrt(self.dim_prev_layer), high=1/np.sqrt(self.dim_prev_layer), 
                                         size=(self.dim_prev_layer, self.dim_layer))
        # self.weights = self.xavier_normal((self.dim_prev_layer, self.dim_layer), self.dim_prev_layer, self.dim_layer)
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

        sum_delta_weights = optimizer.optimization(self.input, loss_gradient, self.d_activation_function)

        return sum_delta_weights
    

'''unit test monk
from Functions import *
from MonkDataProcessing import monk_data
from RegularizationOptimizationClass import Regularization, Optimization

np.random.seed(1)

data = monk_data['training_set_1']
#print(f'dati {data}')
target = monk_data['target_training_set_1']
print(f'target {target}')

layer_hid = Layer(17, 4, activation_functions['leaky_ReLU'], d_activation_functions['d_leaky_ReLU'])
layer_out = Layer(4, 1, activation_functions['sigmoid'], d_activation_functions['d_sigmoid'])

print(f'weights init hidden {layer_hid.weights}')

regulizer = Regularization(reg_type = 'none')

optimizer_hid = Optimization(layer_hid.weights, layer_hid.biases, regulizer, opt_type = 'none')
optimizer_out = Optimization(layer_out.weights, layer_out.biases, regulizer, opt_type = 'none')

accuracy = np.zeros(1000)

for i in range(100):
    correct_classifications = 0
    out_hid = layer_hid.forward_layer(data)
    out_out = layer_out.forward_layer(out_hid)

    d_loss_out = d_loss_functions['d_bce'](target, out_out)
    d_loss_hid = layer_out.backward_layer(d_loss_out, optimizer_out)
    ddd = layer_hid.backward_layer(d_loss_hid, optimizer_hid)

    for k in range(len(out_out)):
        out_out[k] = 1 if out_out[k] >= 0.5 else 0
        if (out_out[k] == target[k]):
            correct_classifications += 1
    accuracy[i] = correct_classifications/len(out_out)
    print(f'accuracy_{i}: {accuracy[i]}')
'''
