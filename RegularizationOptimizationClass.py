import numpy as np

class RegularizationOptimization:
    def __init__(
            self, 
            Lambda_t = 0.5, 
            Lambda_l = 0.5, 
            alpha = 1e-4, 
            learning_rate_w = 1e-4, 
            learning_rate_b = 1e-4, 
            momentum = 0.8, 
            beta_1 = 0.9, 
            beta_2 = 0.999, 
            epsilon = 1e-8, 
            t = 1
            ):
        '''
        Class for regularization and optimization

        Args:
            Lambda_t (float): constant used in Tikhonov regularization.
            Lambda_l (float): constant used in Lasso regularization.
            alpha (float): scale factor for regularization term.
            learning_rate_w (float): growth factor for the Weights parameter of the network.
            learning_rate_b (float): growth factor for the Biases parameter of the network.
            momentum (float): factor for optimization through Nesterov Accelerated Gradient (NAG).
            beta_1 (float): control factor linked to first order momentum for computation of the velocity in Adam Optimization.
            beta_2 (float): control factor linked to second order momentum for computation of the variance in Adam Optimization.
            epsilon (float): stabilization term used in the update of the Weights and the Biases at every step of the Adam Optimization.
            t (int): counter of iterations used in Adam Optimization that goes up to number_epochs * number_batches.
        '''
        self.Lambda_t = Lambda_t
        self.Lambda_l = Lambda_l
        self.alpha = alpha
        self.learning_rate_w = learning_rate_w
        self.learning_rate_b = learning_rate_b
        self.momentum = momentum
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.t = t

    def regularization(self, weights, reg_type):
        '''
        Function that computes the regularization term using Tikhonov, Lasso or Elastic learning rule.

        Args:
            weights (array): it is a placeholder for the kind of weights that are used in the computation of regularization: 
                             in case of NAG Optimization, the predictions of the weights are used; 
                             in case of Adam Optimization, the current weights are used.
            reg_type (string): is the type of Regularization being applied.

        Return: regularization term according to the reg_type being used. To be subtracted to the gradient in the Loss Function.  
        '''
        regularization_terms = {'tikhonov', 'lasso', 'elastic'}
        if reg_type == 'tikhonov':
            reg_term = 2 * self.Lambda_t * weights # learning rule of Tikhonov Regularization
        elif reg_type == 'lasso':
            reg_term = self.Lambda_l * np.sign(weights) # learning rule of Lasso Regularization
        elif reg_type == 'elastic':
            reg_term = (2 * self.Lambda_t * weights + self.Lambda_l * np.sign(weights)) # Tikhonov + Lasso Regularization
        else:
            raise ValueError(f'Invalid {reg_type}. Choose from {', '.join(regularization_terms)}')
        return reg_term
    
'''Unit test'''
np.random.seed(42)
weights = np.random.rand(3, 3)
reg = RegularizationOptimization()
reg_term = reg.regularization(weights, 'lasso')
print(f'weights {weights}')
print(f'reg term {reg_term}')
