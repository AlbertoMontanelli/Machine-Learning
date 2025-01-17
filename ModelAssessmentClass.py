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
            classification_problem = False
    ):
        '''
        Class focused on the actual training and validation of the neural network.

        Args:
            x_retrain (array): data through which the neural network will be re-trained (once 
                               hyperparameters are fixed, after Model Selection).
            target_retrain (array): targets of x_retrain.
            x_test (array): data through which the accuracy of the neural network is estimated.
            target_test (array): targets of x_test.
            epochs (int): number of iterations of the cycle forward propagation + backward propagation). 
            batch_size (int): batch size for training. If batch_size = 1, the neural network is trained using an online learning approach.
                              If batch_size != 1, the neural network is trained using a mini-batch learning approach with batches of size
                              batch_size.
            loss_func (func): loss function.
            d_loss_func (func): derivative of the loss function.
            neural_network (NeuralNetwork): instance of the class NeuralNetwork.
            classification_problem (bool): checking whether the problem consists of regression (default) 
                                           or classification. If classification_problem = True, accuracy
                                           is computed.
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
        self.classification_problem = classification_problem
        '''
        if classification_problem:
            if self.neural_network.layers[-1].activation_function == activation_functions['tanh']:
                self.target_retrain[self.target_retrain == 0] = -1
                self.target_test[self.target_test == 0] = -1
        '''

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


    def retrain_test(
            self         
    ):
        '''
        Function that computes training and test error for each epoch.
        
        Returns:
            retrain_error_tot (array): training error for each epoch.
            test_error_tot (array): test error for each epoch.
            accuracy_retrain_tot (array): accuracy for the training set for each epoch 
                                          (if classification_problem = True).
            accuracy_test_tot (array): accuracy for the test set for each epoch (if classification_problem = True).
                 
        '''
        retrain_error_tot = np.array([])
        test_error_tot = np.array([])

        accuracy_retrain_tot = np.array([])
        accuracy_test_tot = np.array([])

        for i in range(self.epochs):
            retrain_error_epoch, retrain_pred = self.retrain_epoch(self.x_retrain, self.target_retrain)
            retrain_error_tot = np.append(retrain_error_tot, retrain_error_epoch)

            test_error_epoch, test_pred = self.test_epoch(self.x_test, self.target_test)
            test_error_tot = np.append(test_error_tot, test_error_epoch)
            
            if self.classification_problem:
                accuracy_retrain = self.accuracy_curve(retrain_pred, self.target_retrain)
                accuracy_test = self.accuracy_curve(test_pred, self.target_test)

                accuracy_retrain_tot = np.append(accuracy_retrain_tot, accuracy_retrain)
                accuracy_test_tot = np.append(accuracy_test_tot, accuracy_test)


            if ((i + 1) % 10 == 0):
                print(f'epoch {i+1}, retrain error {retrain_error_epoch}, test error {test_error_epoch}')


        if self.classification_problem:
            return retrain_error_tot, test_error_tot, accuracy_retrain_tot, accuracy_test_tot

        else:
            return retrain_error_tot, test_error_tot
