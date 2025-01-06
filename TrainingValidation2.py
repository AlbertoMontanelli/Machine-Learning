import numpy as np
from DataProcessingClass import DataProcessing
from NeuralNetworkClass import NeuralNetwork


class TrainingValidation:

    def __init__(
            self,
            data_splitter,
            epochs,
            batch_size
            ):
        '''
        Class focused on the actual training and validation of the neural network.

        Args:
            data_splitter (DataProcessing): instance of the class DataProcessing.
            epochs (int): number of iterations of the cycle forward propagation + backward propagation + weights update. 
            batch_size (int): batch size for training. If batch_size = 1, the neural network is trained using an online learning approach.
                              If batch_size != 1, the neural network is trained using a mini-batch learning approach with batches of size
                              batch_size.
        '''
        self.data_splitter = data_splitter
        self.epochs = epochs
        self.batch_size = batch_size


    def batch_generator(
            self,
            x_train,
            target_train
            ):
        '''
        Function that splits the training data into mini-batches.

        Args:
            x_train (array): one array of the list of arrays of self.data_splitter.x_trains.
            target_train (array): targets corresponding to self.x_train.

        Returns:
            x_batches (list): list of arrays of data that form the mini-batches.
            target_batches (list): list of arrays of labels corresponding to the data in x_batches.
        '''
        num_samples = x_train.shape[0]
        x_batches = []
        target_batches = []
        for i in range(0, num_samples, self.batch_size):
            batch = x_train[i : i + self.batch_size]
            target_batch = target_train[i : i + self.batch_size]
            x_batches.append(batch)
            target_batches.append(target_batch)
        
        return x_batches, target_batches



'''Unit test for batches'''
np.random.seed(42)
x_tot = np.random.rand(10, 3)
print(f'x_tot pre-shuffle \n {x_tot}')
target_tot = np.random.rand(10, 3)
K = 3

data_split = DataProcessing(x_tot, target_tot, 0, K)
print(f'x_tot \n {data_split.x_trains}')
train_val = TrainingValidation(data_split, 100, 2)
for xx, target in zip(data_split.x_trains, data_split.target_trains):
    batches, target_batches = train_val.batch_generator(xx, target)
    print(f'batches \n {batches}')


