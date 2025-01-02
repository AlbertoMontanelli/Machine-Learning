import numpy as np
import Functions

class Regularization:
    def __init__(
            self, 
            Lambda_t = 0.5, 
            Lambda_l = 0.5, 
            alpha = 1e-4
            ):
        '''
        Class for regularization

        Args:
            Lambda_t (float): constant used in Tikhonov regularization.
            Lambda_l (float): constant used in Lasso regularization.
            alpha (float): scale factor for regularization term.
        '''
        self.Lambda_t = Lambda_t
        self.Lambda_l = Lambda_l
        self.alpha = alpha
        

    def regularization(self, weights, reg_type):
        '''
        Function that computes the regularization term using Tikhonov, Lasso or Elastic learning rule.

        Args:
            weights (array): Weights matrix.
            reg_type (string): the type of Regularization being applied.

        Return: reg_term (float), according to the reg_type being used. To be subtracted to the gradient in the Loss Function.  
        '''

        regularization_type = {'tikhonov', 'lasso', 'elastic'}
        if reg_type == 'tikhonov':
            reg_term = 2 * self.Lambda_t * weights # Learning rule of Tikhonov Regularization
        elif reg_type == 'lasso':
            reg_term = self.Lambda_l * np.sign(weights) # Learning rule of Lasso Regularization
        elif reg_type == 'elastic':
            reg_term = (2 * self.Lambda_t * weights + self.Lambda_l * np.sign(weights)) # Tikhonov + Lasso Regularization
        else:
            raise ValueError(f'Invalid {reg_type}. Choose from {', '.join(regularization_type)}')
        return reg_term


class Optimization:
    def __init__(
            self,
            learning_rate_w = 1e-4, 
            learning_rate_b = 1e-4, 
            momentum = 0.8, 
            beta_1 = 0.9, 
            beta_2 = 0.999, 
            epsilon = 1e-8, 
            t = 1
            ):
        '''
        Class for optimization

        Args:
            learning_rate_w (float): growth factor for the Weights parameter of the network.
            learning_rate_b (float): growth factor for the Biases parameter of the network.
            momentum (float): factor for optimization through Nesterov Accelerated Gradient (NAG).
            beta_1 (float): control factor linked to first order momentum for computation of the velocity in Adam Optimization.
            beta_2 (float): control factor linked to second order momentum for computation of the variance in Adam Optimization.
            epsilon (float): stabilization term used in the update of the Weights and the Biases at every step of the Adam Optimization.
            t (int): counter of iterations used in Adam Optimization that goes up to number_epochs * number_batches.
        '''
        self.learning_rate_w = learning_rate_w
        self.learning_rate_b = learning_rate_b
        self.momentum = momentum
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.t = t
        
    def initialize(self, weights, biases, opt_type):
        '''
        Function that initializes the parameters of the NAG and Adam algorithms.

        Args:
            weights (array): Weights matrix.
            biases (array): Biases array.
            opt_type (string): the type of Optimization being applied.
        '''
        self.weights = weights
        self.biases = biases
        self.opt_type = opt_type
        optimization_type = {'NAG', 'adam'}

        if opt_type == 'NAG':
            # Initialization of the parameters for Nesterov optimization
            self.velocity_weights = np.zeros_like(self.weights)
            self.velocity_biases = np.zeros_like(self.biases)

        elif opt_type == 'adam':
            # Initialization of the parameters for Adam optimization
            self.m_weights = np.zeros_like(self.weights)
            self.v_weights = np.zeros_like(self.weights)
            self.m_biases = np.zeros_like(self.biases)
            self.v_biases = np.zeros_like(self.biases)
        
        else:
            raise ValueError(f'Invalid {self.opt_type}. Choose from {', '.join(optimization_type)}')


    def optimization(self, input, loss_gradient):
        '''
        Function that optimizes the update of the Weights and the Biases using NAG or Adam algorithms.

        Args:
            input (array): input matrix to the current layer.
            loss_gradient (array): derivative of the loss function evaluated in the output values of the network.

        Return: sum_delta_weights (array), loss_gradient for hidden layer   
        '''

        if self.opt_type == 'NAG':
            weights_pred = self.weights + self.momentum * self.velocity_weights  # Predicted weights used to compute the
                                                                                 # gradient after the momentum is applied
            bias_pred = self.biases + self.momentum * self.velocity_biases # Same thing for the biases
            net_pred = np.dot(input, weights_pred) + bias_pred  #  Net computed with respect to the predicted weights and the predicted biases
            delta_pred = - loss_gradient * self.activation_derivative(net_pred)  # Loss gradient with respect to net, minus sign due to the definition
            grad_weights = self.learning_rate_w * np.dot(self.input.T, delta_pred)  # Loss gradient multiplied by the learning rate.
                                                                            # The gradient has been computed with respect to the predicted weights and biases
            
            reg_term = self.regularization_func(Lambda_t, Lambda_l, weights_pred, reg_type)
            self.velocity_weights = momentum * self.velocity_weights + grad_weights - reg_term  # Delta w new 
                                                                                                # the minus sign before reg_term is due to the application of gradient descent algorithm.
            self.weights += self.velocity_weights  # Updating the weights
            self.velocity_biases = momentum * self.velocity_biases + learning_rate_b * np.sum(delta_pred, axis=0, keepdims=True)
            self.biases += self.velocity_biases # Updating the biases


    
'''Unit test for regularization'''
np.random.seed(42)
weights = np.random.rand(3, 3)
reg = RegularizationOptimization()
reg_term = reg.regularization(weights, 'lasso')
print(f'weights {weights}')
print(f'reg term {reg_term}')