import numpy as np

from Functions import activation_functions, d_activation_functions, loss_functions, d_loss_functions
from DataProcessingClass import DataProcessing
from NeuralNetworkClass import NeuralNetwork

class TrainValidation:

    def __init__(self, neural_network, data_split):
        '''
        Class that implements training and validation of the neural network.

        Args:
            neural_network (NeuralNetwork): an instance of the NeuralNetwork class.
            data_split (DataProcessing): an instance of the DataProcessing class.
        '''
        self.data_split = data_split
        self.neural_network = neural_network

    def batch_generator(self, x, target, batch_size):
        '''
        Function that generates data batches to be yielded.

        Args:
            x (array): input data.
            target (array): corresponding labels.
            batch_size (int): size of each batch.

        Returns:
            (array, array): batch of input data and labels.
        '''
        indices = np.arange(x.shape[0])
        x_batch = []
        target_batch = []
        #np.random.shuffle(indices)  # VA SHUFFOLATO ANCHE QUESTO?
        for start in range(0, x.shape[0], batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            x_batch.append(x[batch_indices])
            target_batch.append(target[batch_indices])

        return x_batch, target_batch
    
    def train_epoch(self, x_train, target_train, batch_size, loss_function, loss_function_derivative):
        '''
        Training of the network for a single epoch.

        Args:
            x_train (array): training data.
            target_train (array): training labels.
            batch_size (int): batch size for training. If batch_size = 1, the neural network is trained using an online learning approach.
                              If batch_size != 1, the neural network is trained using a mini-batch learning approach with batches of size
                              batch_size.
            loss_function (func): loss function.
            loss_function_derivative (func): derivative of the loss function.

        Returns:
            float: average training loss for the epoch.
        '''
        total_loss = 0
        x, target = self.batch_generator(x_train, target_train, batch_size)

        for x_batch, target_batch in zip(x, target):
            predictions = self.neural_network.forward(x_batch)
            loss = loss_function(target_batch, predictions)
            total_loss += loss 
            loss_gradient = loss_function_derivative(target_batch, predictions)
            self.neural_network.backward(loss_gradient)
        return total_loss / x_train.shape[0]


    def validate(self, x_val, target_val, loss_function):
        '''
        Validate the network on the validation set.

        Args:
            x_val (array): validation data.
            target_val (array): validation labels.
            loss_function (func): loss function.

        Returns:
            float: validation loss.
        '''
        predictions = self.neural_network.forward(x_val)
        loss = loss_function(target_val, predictions)
        return np.mean(loss)


    def execute(self, epochs, batch_size, loss_function, loss_function_derivative):
        '''
        Executes training and validation using DataProcessing splits.

        Args:
            epochs (int): number of training epochs.
            batch_size (int): batch size for training. If batch_size = 1, the neural network is trained using an online learning approach.
                              If batch_size != 1, the neural network is trained using a mini-batch learning approach with batches of size
                              batch_size.
            loss_function (func): loss function.
            loss_function_derivative (func): derivative of the loss function.

        Returns:
            
        '''

        avg_train_loss = np.zeros(epochs)
        avg_val_loss = np.zeros(epochs)
        # Loop through folds provided by DataProcessing
        for i, (x_train, target_train, x_val, target_val) in enumerate(zip(
            self.data_split.x_trains, 
            self.data_split.target_trains, 
            self.data_split.x_vals, 
            self.data_split.target_vals
        )):
            print(f"Processing Fold {i + 1}/{len(self.data_split.x_trains)}")
            self.neural_network.reinitialize_weights_and_optimizers()  # Reinitialize weights for each fold

            train_losses = []
            val_losses = []

            # Epoch loop
            for epoch in range(epochs):
                train_loss = self.train_epoch(x_train, target_train, batch_size, loss_function, loss_function_derivative)
                val_loss = self.validate(x_val, target_val, loss_function)

                train_losses.append(train_loss)
                val_losses.append(val_loss)

                print(f"Fold {i + 1}, Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        

            avg_train_loss += train_losses
            avg_val_loss += val_losses
        
        avg_val_loss /= (i+1)
        avg_train_loss /= (i+1)
        return avg_train_loss, avg_val_loss

