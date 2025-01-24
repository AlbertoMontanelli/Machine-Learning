import numpy as np

class LossControl:

    def __init__(self, epochs, stopping_patience = 20, overfitting_patience = 10, smooth_patience = 15):
        '''
        Class that implements a mechanism to monitor validation metrics during the training of a model, in order to prematurely
        stop the optimization process.
        
        Args:
            epochs (int): Maximum number of backpropagation iterations that can be executed.
            
        '''
        self.epochs = epochs
        self.stop_count = 0
        self.smooth_count = 0
        self.overfitting_count = 0
        self.stopping_patience = stopping_patience
        self.overfitting_patience = overfitting_patience
        self.smooth_patience = smooth_patience

    
    def stopping_check(self, actual_epoch, val_errors):
        '''
        Function that keeps track of the relative improvement of the validation error. If it remains below a certain threshold
        for 20 consecutive epochs, it returns True.

        Args:
            actual_epoch (int): current epoch of the training.
            val_errors (array): array containing all validation errors from epoch 0 to the current epoch.
            stopping_patience (int): number of epochs to wait for a significant improvement for val_error.

        Return:
            bool: Returns False if the training algorithm should continue;
                  returns True if it should stop.
        '''
        #print(f'stop count: {self.stop_count}')

        perc = actual_epoch/self.epochs

        if perc >= 0.2:
            relative_error_improvement = (val_errors[actual_epoch - 1] - val_errors[actual_epoch]) / val_errors[actual_epoch - 1]
            if (0 <= relative_error_improvement <= 0.0001):
                self.stop_count += 1
                #print(f"early stopping = count: {self.stop_count}. diff: {relative_error_improvement}")
            else:
                self.stop_count = 0

        if self.stop_count >= self.stopping_patience:
            return True
        else:
            return False
    

    def smoothness_check(self, actual_epoch, error_array):
        '''
        Function that checks if the curve is smooth or not.

        Args: 
            actual_epoch (int): current epoch of the training.
            error_array (array): validation or training error array.
            smooth_patience (int): number of epochs for which fluctations of the val_error are tolerated.

        Returns:
            bool: returns False if the curve is not smooth;
                  returns True if it is smooth.
        '''
        perc = actual_epoch/self.epochs

        if perc > 0.2:
            if (error_array[actual_epoch]-error_array[actual_epoch - 1])/ error_array[actual_epoch - 1] > 0.001:
                self.smooth_count += 1
            
            if self.smooth_count >= self.smooth_patience:
                return False
            else:
                return True
            
    def overfitting_check(self, actual_epoch, train_error, val_error):
        '''
        Function that checks the overfitting, respectively checking whether the validation error rises, or the velocities
        of the loss function computed for the validation and the training error vary too differently, or if the difference 
        between training error and validation error for successive epochs is above a certain threshold.

        Args:
            actual_epoch (int): current epoch of the training.
            train_error (array): array of the training errors.
            val_error (array): array of the validation errors.
            overfitting_patience (int): number of epochs for which the risk of overfitting is tolerated.

        Returns:
            bool: returns True if there is overfitting,
                  False if there is no overfitting.
        '''
        perc = actual_epoch/self.epochs
        #diff_act = (val_error[actual_epoch] - train_error[actual_epoch]) - (val_error[actual_epoch - 1] - train_error[actual_epoch - 1])
        #diff_prec = (val_error[actual_epoch - 50] - train_error[actual_epoch - 50]) - (val_error[actual_epoch - 51] - train_error[actual_epoch - 51])
        #relative_distance = ((val_error[actual_epoch] - train_error[actual_epoch]) - (val_error[actual_epoch - 1] - train_error[actual_epoch - 1]))/(val_error[actual_epoch - 1] - train_error[actual_epoch - 1])
        if perc > 0.2:
            if ((val_error[actual_epoch] - val_error[actual_epoch - 1] >= 0) 
                #or 
                #((train_error[actual_epoch] - train_error[actual_epoch - 1]) / (val_error[actual_epoch] - val_error[actual_epoch - 1]) > 2) # train speed = 2 * val speed
                #or
                #((relative_distance > 0.1) and (val_error[actual_epoch] > train_error[actual_epoch]))
                #or 
                #((diff_act > diff_prec) and (val_error[actual_epoch] > train_error[actual_epoch]))
                ):
                    self.overfitting_count += 1
                    #print(f"overfitting = count: {self.overfitting_count}, actual epoch: {actual_epoch}, relative distance: {diff_act - diff_prec}, diff: {val_error[actual_epoch] - val_error[actual_epoch - 1]}")
            else:
                self.overfitting_count = 0

            if self.overfitting_count >= self.overfitting_patience:
                return True
            else:
                return False