import numpy as np

from Functions import activation_functions

class ModelAssessment:

    def __init__(
            self,
            x_retrain,
            target_retrain,
            x_test,
            target_test,
            epochs,
            batch_size,
            loss_func,
            d_loss_func,
            neural_network,
            loss_control,
            classification_problem = False
    ):
        '''
        Class focused on the actual training and validation of the neural network.

        Args:
            x_retrain (array): data through which the neural network will be re-trained (once hyperparameters are fixed, 
                               after Model Selection).
            target_retrain (array): targets of x_retrain.
            x_test (array): data through which the performace of the neural network is evaluated.
            target_test (array): targets of x_test.
            epochs (int): number of iterations of the cycle (forward propagation + backward propagation). 
            batch_size (int): batch size for training. 
                              If batch_size = 1, the neural network is trained using an online learning approach.
                              If 1 < batch_size < len(x_retrain), the neural network is trained using a mini-batch learning approach with
                              batches of size batch_size.
                              If batch_size = len(x_retrain), the neural network is trained using a batch learning approch.
            loss_func (func): loss function.
            d_loss_func (func): derivative of the loss function.
            neural_network (NeuralNetwork): instance of the class NeuralNetwork. The forward() and backward() methods are used.
                In particular, forward() returns:
                    pred (array): array containing the outputs of the neural network for the given input data.
            loss_control (LossControl): instance of the class LossControl. The methods stopping_check(), smoothness_check() and 
                                        overfitting_check() are being used.
                Return, respectively:
                    early_stopping: is True if the training is being early-stopped, 
                                    is False if it continues.
                    smoothness: is True if the loss function is smooth, 
                                is False if the loss function is not smooth.
                    overfitting: is True if there overfitting,
                                 is False if there is not.
            classification_problem (bool): checking whether the problem consists of regression (default) or classification. 
                                           If classification_problem is True, accuracy is computed.
        '''

        self.x_retrain = x_retrain
        self.target_retrain = target_retrain
        self.x_test = x_test
        self.target_test = target_test
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_func = loss_func
        self.d_loss_func = d_loss_func
        self.neural_network = neural_network
        self.loss_control = loss_control
        self.classification_problem = classification_problem


    def batch_generator( 
            self,
            xx,
            target
    ):
        '''
        Function that splits the training data into mini-batches of size self.batch_size.

        Args:
            xx (array): data to be split.
            target (array): targets of xx.

        Returns:
            x_batches (list): list of arrays of data that form the mini-batches.
            target_batches (list): list of arrays of labels corresponding to the data in x_batches.
        '''

        num_samples = xx.shape[0]
        if self.batch_size > num_samples:
            raise ValueError(f'Invalid batch size {self.batch_size}. Must be smaller than number of examples {num_samples}')
        x_batches = []
        target_batches = []
        for i in range(0, num_samples, self.batch_size):
            batch = xx[i : i + self.batch_size]
            target_batch = target[i : i + self.batch_size]
            x_batches.append(batch)
            target_batches.append(target_batch)
        
        return x_batches, target_batches
    
    def retrain_epoch(
            self,
            xx,
            target
    ):
        '''
        Function that computes the average training error during the re-training of the network for a single epoch
        and returns its output array.
        
        Args:
            xx (array): data through which the re-training is being performed.
            target (array): targets of xx.
            
        Returns:
            train_error_epoch (float): average training error of one epoch. 
            prediction_retrain (array): array of the outputs of the neural network for the training data.
        '''
       
        train_error_epoch = 0
        prediction_retrain = np.array([])
        batches, target_batches = self.batch_generator(xx, target)

        for batch, target_batch in zip(batches, target_batches):

            pred = self.neural_network.forward(batch)
            prediction_retrain = np.append(prediction_retrain, pred)
            train_error_epoch += self.loss_func(target_batch, pred)
            d_loss = self.d_loss_func(target_batch, pred)
            self.neural_network.backward(d_loss)

        train_error_epoch /= xx.shape[0]

        return train_error_epoch, prediction_retrain
    
    
    def test_epoch(
            self,
            xx,
            target
    ):
        '''
        Function that computes the average test error for a single epoch and returns its output array.

        Args:
            xx (array): test data.
            target (array): targets of xx.       

        Returns:
            test_error_epoch (float): average test error of one epoch.
            prediction_test (array): array of the outputs of the neural network for the test data.
        '''

        prediction_test = self.neural_network.forward(xx)
        test_error_epoch = self.loss_func(target, prediction_test)/xx.shape[0]

        return test_error_epoch, prediction_test

    
    def accuracy_curve(
            self,
            prediction,
            target
    ): 
        '''
        Function that computes the accuracy-per-epoch.

        Args:
            prediction (array): output of the network.
            target (array): targets corresponding to the data for which prediction has been computed.

        Returns: 
            accuracy (float): accuracy computed for one epoch
        '''

        correct_classifications = 0

        for k in range(len(prediction)):
            if (self.neural_network.layers[-1].activation_function == activation_functions['sigmoid']):
                prediction[k] = 1 if prediction[k] >= 0.5 else 0
            elif (self.neural_network.layers[-1].activation_function == activation_functions['tanh']):
                prediction[k] = 1 if prediction[k] > 0 else 0
            else:
                raise ValueError("Invalid activation function for the output layer for classification task. Choose between sigmoid or tanh")

            if (prediction[k] == target[k]):
                correct_classifications += 1

        accuracy = correct_classifications/len(prediction)

        return accuracy

    
    def loss_control_epoch(
            self, 
            epoch,
            train_error,
            val_error,
            early_stopping,
            smoothness,
            overfitting
    ):
        '''
        Function that checks the goodness of the average of the loss functions.

        Args:
            epoch (int): current epoch.
            train_error (array): array of the error-per-epoch of the training set.
            val_error (array): array of the error-per-epoch of the validation set.
            early_stopping (bool): inherited from train_fold.
            smoothness (bool): inherited from train_fold.
            overfitting (bool): inherited from train_fold.

        Returns:
            smoothness_check (bool): True if the curve is smooth, False if it is not smooth.
            stop_epoch (bool): True if the training process has been stopped by either early_stopping
                               or overfitting, False if it has not been stopped.
        '''

        stop_epoch = epoch
        
        if overfitting:
            overfitting_check = self.loss_control.overfitting_check(epoch, val_error)
            if overfitting_check:
                print(f"Overfitting at epoch {epoch}")
                stop_epoch = epoch - self.loss_control.overfitting_patience 

        if early_stopping:
            early_check = self.loss_control.stopping_check(epoch, val_error)
            if early_check:
                print(f"Early stopping at epoch {epoch}")
                stop_epoch = epoch - self.loss_control.stopping_patience

        if smoothness:
            smoothness_check_train = self.loss_control.smoothness_check(epoch, train_error)
            
            if (smoothness_check_train == False):
                smoothness_check = False
            else:
                smoothness_check = True

        if smoothness:
            return smoothness_check, stop_epoch
        else:
            return stop_epoch                    


    def retrain_test(
            self,
            early_stopping = False,
            smoothness = False,
            overfitting = False    
    ):
        '''
        Function that computes training and test error for each epoch.

        Args:
            early_stopping (bool): False by default, if True enables overfitting checking.
            smoothness (bool): False by default, if True enables smoothness checking.
            overfitting (bool): False by default, if True enables overfitting checking.
        
        Returns:
            retrain_error_tot (array): training error for each epoch.
            test_error_tot (array): test error for each epoch.
            If classification_problem == True returns also:
                accuracy_retrain_tot (array): accuracy for the training set for each epoch.
                accuracy_test_tot (array): accuracy for the test set for each epoch.
        '''

        retrain_error_tot = []
        test_error_tot = []

        accuracy_retrain_tot = []
        accuracy_test_tot = []

        for epoch in range(self.epochs):
            retrain_error_epoch, retrain_pred = self.retrain_epoch(self.x_retrain, self.target_retrain)
            retrain_error_tot.append(retrain_error_epoch)

            test_error_epoch, test_pred = self.test_epoch(self.x_test, self.target_test)
            test_error_tot.append(test_error_epoch)
            
            if self.classification_problem:
                accuracy_retrain = self.accuracy_curve(retrain_pred, self.target_retrain)
                accuracy_test = self.accuracy_curve(test_pred, self.target_test)

                accuracy_retrain_tot.append(accuracy_retrain)
                accuracy_test_tot.append(accuracy_test)

            if smoothness or early_stopping or overfitting:
                if smoothness:
                    smoothness_outcome, stop_epoch = self.loss_control_epoch(epoch, retrain_error_tot, test_error_tot, early_stopping, smoothness, overfitting)
                    if(smoothness_outcome) == False:
                        print("Function is not smooth")
                        break
                    if(stop_epoch < epoch):
                        break
                else:
                    stop_epoch = self.loss_control_epoch(epoch, retrain_error_tot, test_error_tot, early_stopping, smoothness, overfitting)
                
            # if ((epoch + 1) % 50 == 0):
            #     print(f'epoch {epoch+1}, retrain error {retrain_error_epoch}, test error {test_error_epoch}')


        if smoothness or early_stopping or overfitting:
            #retrain_error_tot = retrain_error_tot[:stop_epoch]
            #test_error_tot = test_error_tot[:stop_epoch]
            self.loss_control.stop_count = 0
            self.loss_control.smooth_count = 0
            self.loss_control.overfitting_count = 0
            if smoothness:
                print(f'smoothness: {smoothness_outcome}')

        print(f'Last retrain error: {retrain_error_tot[-1]}')
        print(f'Last test error: {test_error_tot[-1]}')

        if self.classification_problem:
            return retrain_error_tot, test_error_tot, accuracy_retrain_tot, accuracy_test_tot

        else:
            return retrain_error_tot, test_error_tot
