import numpy as np

class EarlyStopping:

    def __init__(self, epochs):
        '''
        Class that implements a mechanism to monitor validation metrics during the training of a model, in order to prematurely
        stop the optimization process.
        
        Args:
            epochs (int): Maximum number of backpropagation iterations that can be executed.
            
        '''
        self.epochs = epochs
        self.stop_count = 0
        self.smooth_count = 0

    
    def stopping_check(self, actual_epoch, val_errors):
        '''
        Function that keep tracks of the relative improvement of the validation error, and if it remains below a certain threshold
        for 20 consecutive epochs, it returns True.

        Args:
            actual_epoch (int): current epoch of the training.
            val_errors (array): array containing all validation errors from epoch 0 to the current epoch.

        Return:
            bool: Return False if the training algorithm should continue;
                  return True if it should stop.
        '''
        self.actual_epoch = actual_epoch

        self.perc = self.actual_epoch/self.epochs

        if self.perc >= 0.2:
            relative_error_improvement = (val_errors[actual_epoch - 2] - val_errors[actual_epoch - 1]) / val_errors[actual_epoch - 2]
            if relative_error_improvement <= 0.001:
                self.stop_count += 1
                # print(f"count: {self.stop_count}. diff: {relative_error_improvement}")
            else:
                self.stop_count = 0

        if self.stop_count >= 20:
            return True
        else:
            return False
    

    def smoothness_check(self, error_array):
        '''
        Function that check if the curve is smooth or not.

        Args: 
            error_array (array): validation or training error array

        Returns:
            bool: Returns False if the curve is not smooth;
                  return True if it is smooth.
        '''
        
        if self.perc > 0.2:
            if error_array[self.actual_epoch] > error_array[self.actual_epoch - 1]:
                self.smooth_count += 1
            
            if self.smooth_count > 10:
                return False
            else:
                return True    