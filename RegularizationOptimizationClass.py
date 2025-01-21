import numpy as np

class Regularization:

    def __init__(
            self,  
            Lambda = 1e-4,
            alpha = 0.5,
            reg_type = 'elastic'
            ):
        '''
        Class for regularization

        Args:
            Lambda (float): constant used in elastic regularization.
            alpha (float): parameter used in elastic regularization to tune between Tikhonov and Lass regularization.             
            reg_type (str): the type of Regularization being applied.
        '''
        self.Lambda = Lambda
        self.alpha = alpha
        self.reg_type = reg_type

        if not(0<= self.alpha <=1):
            raise ValueError(f"Invalid value for alpha: {self.alpha}. It must be between 0 and 1")
        

    def regularization(
            self, 
            weights
            ):
        '''
        Function that computes the regularization term using Tikhonov, Lasso or Elastic learning rule.

        Args:
            weights (array): weights matrix.

        Returns: 
            reg_term (float): Result of the computation of the regularization algorithm.
                              To be subtracted to the gradient in the Loss Function.  
        '''
        regularization_type = {'tikhonov', 'lasso', 'elastic', 'none'} # The only types of regularization accepted
 
        if self.reg_type == 'tikhonov':
            self.alpha = 0 
        elif self.reg_type == 'lasso':
            self.alpha = 1 
        elif self.reg_type == 'elastic':
            pass
        elif self.reg_type == 'none':
            self.Lambda = 0 # No regularization
        else:
            raise ValueError(f"Invalid {self.reg_type}. Choose from {', '.join(regularization_type)}")
        
        reg_term = self.Lambda * (2  * (1-self.alpha) * weights + self.alpha * np.sign(weights)) # Learning rule of Elastic Regularization.
                                                                                                 # If self.alpha == 0: Tikhonov Regularization.
                                                                                                 # If self.alpha == 1: Lasso Regularizaion.

        return reg_term


class Optimization:

    def __init__(
            self,
            weights,
            biases,
            regulizer,
            opt_type,
            learning_rate = 1e-2,
            momentum = 0.8, 
            beta_1 = 0.9, 
            beta_2 = 0.999, 
            epsilon = 1e-8, 
            t = 1
            ):
        '''
        Class for optimization

        Args:
            weights (array): weights matrix.
            biases (array): biases array.
            regulizer (Regularization): instance of the Regularization class.
            opt_type (str): the type of Optimization being applied.
            learning_rate (float): growth factor for the weights and biases parameters of the network.
            momentum (float): factor for optimization through Nesterov Accelerated Gradient (NAG).
            beta_1 (float): control factor linked to first order momentum for computation of the velocity in Adam Optimization.
            beta_2 (float): control factor linked to second order momentum for computation of the variance in Adam Optimization.
            epsilon (float): stabilization term used in the update of the weights and the biases at every step of the Adam Optimization.
            t (int): counter of iterations used in Adam Optimization that goes up to number_epochs * number_batches.
        '''
        
        self.regulizer = regulizer
        self.opt_type = opt_type
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.t = t
        self.initialization(weights, biases)
        
        
    def initialization(
            self,
            weights,
            biases
            ):
        '''
        Function that initializes the parameters of the NAG and Adam algorithms.

        Args:
            weights (array): weights matrix.
            biases (array): biases array.
        '''
        self.weights = weights
        self.biases = biases
        optimization_type = {'NAG', 'adam', 'none'} # The only types of optimization accepted

        if self.opt_type == 'NAG':
            # Initialization of the parameters for Nesterov optimization
            self.velocity_weights = np.zeros_like(self.weights)
            self.velocity_biases = np.zeros_like(self.biases)

        elif self.opt_type == 'adam':
            # Initialization of the parameters for Adam optimization
            self.m_weights = np.zeros_like(self.weights)
            self.v_weights = np.zeros_like(self.weights)
            self.m_biases = np.zeros_like(self.biases)
            self.v_biases = np.zeros_like(self.biases)
        
        elif self.opt_type == 'none':
            pass

        else:
            raise ValueError(f"Invalid {self.opt_type}. Choose from {', '.join(optimization_type)}")


    def optimization(
            self,
            input,
            loss_gradient,
            d_activation_function
            ):
        '''
        Function that optimizes the update of the Weights and the Biases using NAG or Adam algorithms.

        Args:
            input (array): input matrix to the current layer.
            loss_gradient (array): derivative of the loss function evaluated in the output values of the network.
            d_activation_function (func): derivative of the activation function evaluated in net.

        Returns:
            sum_delta_weights (array): loss_gradient for hidden layer   
        '''
        delta = loss_gradient * d_activation_function(np.dot(input, self.weights) + self.biases) # used for adam and none optimization

        if self.opt_type == 'NAG':
            # Updating weights and biases using NAG optimization
            weights_pred = self.weights + self.momentum * self.velocity_weights  # Predicted weights used to compute the
                                                                                 # gradient after the momentum is applied
            bias_pred = self.biases + self.momentum * self.velocity_biases # Same thing for the biases
            net_pred = np.dot(input, weights_pred) + bias_pred  # Net computed with respect to the predicted weights and 
                                                                # the predicted biases
            delta_pred = loss_gradient * d_activation_function(net_pred)  # Loss gradient with respect to predicted net, 
            grad_weights = self.learning_rate * np.dot(input.T, delta_pred) / len(input)  # Loss gradient multiplied by the learning rate.
                                                                             # The gradient has been computed with respect
                                                                             # to the predicted weights and biases
            reg_term = self.regulizer.regularization(weights_pred)

            # Difference between the current weights and the previous weights. 
            # The minus sign before reg_term and grad_weights is due to the application of gradient descent algorithm
            self.velocity_weights = self.momentum * self.velocity_weights - grad_weights - reg_term 
            self.velocity_biases = self.momentum * self.velocity_biases - self.learning_rate * (np.sum(delta_pred, axis=0, keepdims=True)) / len(input)
            
            self.weights += self.velocity_weights  # Updating the weights
            self.biases += self.velocity_biases # Updating the biases

        elif self.opt_type == 'adam':
            # Updating weights and biases using adam optimization
            reg_term = self.regulizer.regularization(self.weights)

            # np.dot(input.T, self.delta) is dLoss/dw. 
            self.m_weights = self.beta_1 * self.m_weights + (1 - self.beta_1) * (np.dot(input.T, delta) / len(input) + reg_term)
            self.v_weights = self.beta_2 * self.v_weights + (1 - self.beta_2) * ((np.dot(input.T, delta) / len(input) + reg_term)**2) 

            m_weights_hat = self.m_weights / (1 - self.beta_1**self.t)
            v_weights_hat = self.v_weights / (1 - self.beta_2**self.t)

            self.m_biases = self.beta_1 * self.m_biases + (1 - self.beta_1) * np.sum(delta, axis=0, keepdims=True) / len(input)
            self.v_biases = self.beta_2* self.v_biases + (1 - self.beta_2) * np.sum(delta**2, axis=0, keepdims=True) / len(input)**2

            m_biases_hat = self.m_biases / (1 - self.beta_1**self.t)
            v_biases_hat = self.v_biases / (1 - self.beta_2**self.t)

            # Update of weights and biases
            self.weights -= self.learning_rate * m_weights_hat / (np.sqrt(v_weights_hat) + self.epsilon)
            self.biases -= self.learning_rate * m_biases_hat / (np.sqrt(v_biases_hat) + self.epsilon)

        else:
            # Updating weights and biases without using optimization 
            reg_term = self.regulizer.regularization(self.weights)
            self.weights -= (self.learning_rate * np.dot(input.T, delta) / len(input) + reg_term)
            self.biases -= np.sum(delta, axis=0, keepdims=True)/len(input)

        sum_delta_weights = np.dot(delta, self.weights.T) # loss gradient for hidden layer
        return sum_delta_weights

# POSSIBILI MIGLIORAMENTI PER LA CLASSE

# class Optimizer:
#     def __init__(self, weights, biases, learning_rate_w, learning_rate_b, regularizer):
#         self.weights = weights
#         self.biases = biases
#         self.learning_rate_w = learning_rate_w
#         self.learning_rate_b = learning_rate_b
#         self.regularizer = regularizer

#     def update(self, input, loss_gradient, d_activation_function):
#         raise NotImplementedError("Update method must be implemented by subclasses.")

# class NAGOptimizer(Optimizer):
#     def __init__(self, weights, biases, learning_rate_w, learning_rate_b, regularizer, momentum):
#         super().__init__(weights, biases, learning_rate_w, learning_rate_b, regularizer)
#         self.momentum = momentum
#         self.velocity_weights = np.zeros_like(weights)
#         self.velocity_biases = np.zeros_like(biases)

#     def update(self, input, loss_gradient, d_activation_function):
#         # Logica per NAG...
#         pass

# class AdamOptimizer(Optimizer):
#     def __init__(self, weights, biases, learning_rate_w, learning_rate_b, regularizer, beta_1, beta_2, epsilon, t):
#         super().__init__(weights, biases, learning_rate_w, learning_rate_b, regularizer)
#         self.beta_1 = beta_1
#         self.beta_2 = beta_2
#         self.epsilon = epsilon
#         self.t = t
#         self.m_weights = np.zeros_like(weights)
#         self.v_weights = np.zeros_like(weights)

#     def update(self, input, loss_gradient, d_activation_function):
#         # Logica per Adam...
#         pass
