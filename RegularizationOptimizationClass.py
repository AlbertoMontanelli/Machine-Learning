import numpy as np

class Regularization:

    def __init__(
            self, 
            Lambda_t = 0.5, 
            Lambda_l = 0.5, 
            alpha = 1e-4,
            reg_type = 'elastic'
            ):
        '''
        Class for regularization

        Args:
            Lambda_t (float): constant used in Tikhonov regularization.
            Lambda_l (float): constant used in Lasso regularization.
            alpha (float): scale factor for regularization term.
            reg_type (str): the type of Regularization being applied.
        '''
        self.Lambda_t = Lambda_t
        self.Lambda_l = Lambda_l
        self.alpha = alpha
        self.reg_type = reg_type
        

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
            reg_term = 2 * self.Lambda_t * weights # Learning rule of Tikhonov Regularization
        elif self.reg_type == 'lasso':
            reg_term = self.Lambda_l * np.sign(weights) # Learning rule of Lasso Regularization
        elif self.reg_type == 'elastic':
            reg_term = (2 * self.Lambda_t * weights + self.Lambda_l * np.sign(weights)) # Tikhonov + Lasso Regularization
        elif self.reg_type == 'none':
            reg_term = 0 # No regularization
        else:
            raise ValueError(f"Invalid {self.reg_type}. Choose from {', '.join(regularization_type)}")
        return reg_term


class Optimization:

    def __init__(
            self,
            weights,
            biases,
            regulizer,
            opt_type,
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
            weights (array): weights matrix.
            biases (array): biases array.
            regulizer (Regularization): instance of the Regularization class.
            opt_type (str): the type of Optimization being applied.
            learning_rate_w (float): growth factor for the Weights parameter of the network.
            learning_rate_b (float): growth factor for the Biases parameter of the network.
            momentum (float): factor for optimization through Nesterov Accelerated Gradient (NAG).
            beta_1 (float): control factor linked to first order momentum for computation of the velocity in Adam Optimization.
            beta_2 (float): control factor linked to second order momentum for computation of the variance in Adam Optimization.
            epsilon (float): stabilization term used in the update of the Weights and the Biases at every step of the Adam Optimization.
            t (int): counter of iterations used in Adam Optimization that goes up to number_epochs * number_batches.
        '''
        self.regulizer = regulizer
        self.learning_rate_w = learning_rate_w
        self.learning_rate_b = learning_rate_b
        self.momentum = momentum
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.t = t
        self.opt_type = opt_type
        self.initialization(weights, biases)


    def initialization(self, weights, biases):
        '''
        Function that initializes the parameters of the NAG and Adam algorithms.

        Args:
            weights (array): weights matrix.
            biases (array): biases array.
        '''
        self.weights = weights
        self.biases = biases
        optimization_type = {'NAG', 'adam'} # The only types of optimization accepted

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
        
        else:
            raise ValueError(f"Invalid {self.opt_type}. Choose from {', '.join(optimization_type)}")


    def optimization(self, input, loss_gradient, d_activation_function):
        '''
        Function that optimizes the update of the Weights and the Biases using NAG or Adam algorithms.

        Args:
            input (array): input matrix to the current layer.
            loss_gradient (array): derivative of the loss function evaluated in the output values of the network.
            d_activation_function (func): derivative of the activation function evaluated in net.

        Returns:
            sum_delta_weights (array): loss_gradient for hidden layer   
        '''
        self.delta = - loss_gradient * d_activation_function(np.dot(input, self.weights) + self.biases)

        if self.opt_type == 'NAG':
            weights_pred = self.weights + self.momentum * self.velocity_weights  # Predicted weights used to compute the
                                                                                 # gradient after the momentum is applied
            bias_pred = self.biases + self.momentum * self.velocity_biases # Same thing for the biases
            net_pred = np.dot(input, weights_pred) + bias_pred  # Net computed with respect to the predicted weights and 
                                                                # the predicted biases
            delta_pred = - loss_gradient * d_activation_function(net_pred)  # Loss gradient with respect to net, 
                                                                                  # minus sign due to the definition
            grad_weights = self.learning_rate_w * np.dot(input.T, delta_pred)  # Loss gradient multiplied by the learning rate.
                                                                               # The gradient has been computed with respect
                                                                               # to the predicted weights and biases
            
            reg_term = self.regulizer.regularization(weights_pred)

            # Difference between the current weights and the previous weights. 
            # The minus sign before reg_term is due to the application of gradient descent algorithm
            self.velocity_weights = self.momentum * self.velocity_weights + grad_weights - self.regulizer.alpha * reg_term  
            self.weights += self.velocity_weights  # Updating the weights
            self.velocity_biases = self.momentum * self.velocity_biases + self.learning_rate_b * np.sum(delta_pred, axis=0, keepdims=True)
            self.biases += self.velocity_biases # Updating the biases

        else:
            reg_term = self.regulizer.regularization(self.weights)

            # np.dot(input.T, delta) is dLoss/dw. 
            # Since self.delta is defined with a minus sign and the formula is with a plus sign, we put a minus sign in front of np.dot()
            self.m_weights = self.beta_1 * self.m_weights + (1 - self.beta_1) * (- np.dot(input.T, self.delta) - reg_term)
            # here we have a plus sign in front of (1 - self.beta_2) since self.delta is squared
            self.v_weights = self.beta_2* self.v_weights + (1 - self.beta_2) * ((- np.dot(input.T, self.delta) - reg_term)**2) 
            m_weights_hat = self.m_weights / (1 - self.beta_1**self.t)
            v_weights_hat = self.v_weights / (1 - self.beta_2**self.t)

            self.m_biases = self.beta_1 * self.m_biases - (1 - self.beta_1) * np.sum(self.delta, axis=0, keepdims=True)
            self.v_biases = self.beta_2* self.v_biases + (1 - self.beta_2) * np.sum(self.delta**2, axis=0, keepdims=True)
            m_biases_hat = self.m_biases / (1 - self.beta_1**self.t)
            v_biases_hat = self.v_biases / (1 - self.beta_2**self.t)

            self.weights -= self.learning_rate_w * m_weights_hat / (np.sqrt(v_weights_hat) + self.epsilon)
            self.biases -= self.learning_rate_b * m_biases_hat / (np.sqrt(v_biases_hat) + self.epsilon)

        sum_delta_weights = np.dot(self.delta, self.weights.T) # loss gradient for hidden layer
        return sum_delta_weights
