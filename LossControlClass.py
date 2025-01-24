class LossControl:


    def __init__(self, epochs, stopping_patience = 20, overfitting_patience = 10, smooth_patience = 15):
        '''
        Class that implements mechanisms to monitor validation metrics during the training of a model, 
        in order to stop the optimization process before reaching the number of epochs stabilized a priori.
        
        Args:
            epochs (int): maximum number of training iterations that can be executed.
            stopping_patience (int): number of epochs to wait for a significant improvement for val_error.
            overfitting_patience (int): number of epochs for which the risk of overfitting is tolerated.
            smooth_patience (int): number of epochs for which fluctations of the val_error are tolerated.   
        '''
        self.epochs = epochs
        self.stop_count = 0
        self.smooth_count = 0
        self.overfitting_count = 0
        self.stopping_patience = stopping_patience
        self.overfitting_patience = overfitting_patience
        self.smooth_patience = smooth_patience
        if not (isinstance(stopping_patience, int) and stopping_patience > 0):
            raise ValueError(f'Invalid value for stopping patience {stopping_patience}. Choose a positive integer.')
        if not (isinstance(overfitting_patience, int) and overfitting_patience > 0):
            raise ValueError(f'Invalid value for overfitting patience {overfitting_patience}. Choose a positive integer.')
        if not (isinstance(smooth_patience, int) and smooth_patience > 0):
            raise ValueError(f'Invalid value for smoothness patience {smooth_patience}. Choose a positive integer.')

    
    def stopping_check(self, current_epoch, val_errors):
        '''
        Function that keeps track of the relative improvement of the validation error. 
        If it remains below a certain threshold for a number of epochs larger than the
        stopping patience, it returns True.

        Args:
            current_epoch (int): current epoch of the training.
            val_errors (array): array containing all validation errors from epoch 0 to the current epoch.

        Returns:
            bool: Returns False if the training algorithm should continue;
                  returns True if it should stop.
                  Bool is used in ModelSelection.loss_control_avg().
        '''
        perc = current_epoch/self.epochs

        if perc >= 0.2: # stage of the training after which the check is performed
            relative_error_improvement = (val_errors[current_epoch - 1] - val_errors[current_epoch]) / val_errors[current_epoch - 1]
            if (0 <= relative_error_improvement <= 0.0001):
                self.stop_count += 1
            else:
                self.stop_count = 0 # we want the epochs in which there is little to no improvement to be consecutive

        if self.stop_count >= self.stopping_patience:
            return True
        else:
            return False
    

    def smoothness_check(self, current_epoch, error_array): 
        ''' 
        Function that checks if the curve is smooth or not.
 
        Args:  
            current_epoch (int): current epoch of the training. 
            error_array (array): validation or training error array. 
           
        Returns: 
            bool: False if the curve is not smooth; 
                  True if it is smooth. 
                  Bool is used in ModelSelection.loss_control_avg().
        ''' 
        perc = current_epoch/self.epochs 
 
        if perc > 0.2: 
            control = (error_array[current_epoch] - error_array[current_epoch - 1]) * (error_array[current_epoch - 1] - error_array[current_epoch - 2]) 
            if control < 0: 
                self.smooth_count += 1 
             
            if self.smooth_count >= self.smooth_patience: 
                return False 
            else: 
                return True
            
    def overfitting_check(self, current_epoch, val_error):
        '''
        Function that checks the overfitting, respectively checking whether the validation error rises, or the velocities
        of the loss function computed for the validation and the training error vary too differently, or if the difference 
        between training error and validation error for successive epochs is above a certain threshold.

        Args:
            current_epoch (int): current epoch of the training.
            train_error (array): array of the training errors.
            val_error (array): array of the validation errors.
            

        Returns:
            bool: returns True if there is overfitting,
                  False if there is no overfitting.
                  Bool is used in ModelSelection.loss_control_avg().
        '''
        perc = current_epoch/self.epochs
        #diff_act = (val_error[current_epoch] - train_error[current_epoch]) - (val_error[current_epoch - 1] - train_error[current_epoch - 1])
        #diff_prec = (val_error[current_epoch - 50] - train_error[current_epoch - 50]) - (val_error[current_epoch - 51] - train_error[current_epoch - 51])
        #relative_distance = ((val_error[current_epoch] - train_error[current_epoch]) - (val_error[current_epoch - 1] - train_error[current_epoch - 1]))/(val_error[current_epoch - 1] - train_error[current_epoch - 1])
        if perc > 0.2:
            if ((val_error[current_epoch] - val_error[current_epoch - 1] >= 0) 
                #or 
                #((train_error[current_epoch] - train_error[current_epoch - 1]) / (val_error[current_epoch] - val_error[current_epoch - 1]) > 2) # train speed = 2 * val speed
                #or
                #((relative_distance > 0.1) and (val_error[current_epoch] > train_error[current_epoch]))
                #or 
                #((diff_act > diff_prec) and (val_error[current_epoch] > train_error[current_epoch]))
                ):
                    self.overfitting_count += 1
                    #print(f"overfitting = count: {self.overfitting_count}, actual epoch: {current_epoch}, relative distance: {diff_act - diff_prec}, diff: {val_error[current_epoch] - val_error[current_epoch - 1]}")
            else:
                self.overfitting_count = 0

            if self.overfitting_count >= self.overfitting_patience:
                return True
            else:
                return False