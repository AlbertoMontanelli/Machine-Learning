import numpy as np

class EarlyStopping:

    def __init__(self, epochs):
        '''
        
        Args:
            epochs: 
            
        '''
        self.epochs = epochs
        self.stop_count = 0
        self.smooth_count = 0

    
    def stopping_check(self, actual_epoch, val_errors):
        '''
        Args:
            actual_epoch (int):
            val_errors (array): array containing all validation errors from epoch 0 to the current epoch.

        Return:
            bool
        '''
        self.actual_epoch = actual_epoch

        self.perc = self.actual_epoch/self.epochs

        if self.perc >= 0.4:
            relative_error_improvement = (val_errors[actual_epoch - 1] - val_errors[actual_epoch]) / val_errors[actual_epoch - 1]
            if relative_error_improvement <= 0.001:
                self.stop_count += 1
                # print(f"count: {self.stop_count}. diff: {relative_error_improvement}")
            else:
                self.stop_count = 0

        if self.stop_count == 20:
            return True
        else:
            return False
    

    def smoothness_check(self, error_array):
        '''
        Function that check if the curve is smooth or not

        Args: 
            error_array (array): val o training error array

        Returns:
            bool: True if the curve isn't smooth, False if the curve is smooth
        '''
        if self.perc > 0.1:
            if error_array[self.actual_epoch] > error_array[self.actual_epoch - 1]:
                return True
            else:
                return False



   





    
    