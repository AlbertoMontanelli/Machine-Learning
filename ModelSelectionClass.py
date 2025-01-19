import numpy as np

class ModelSelection:

    def __init__(
            self,
            data_splitter,
            epochs,
            batch_size,
            loss_func,
            d_loss_func,
            neural_network,
            early_stop
    ):
        '''
        Class focused on the actual training and validation of the neural network.

        Args:
            data_splitter (DataProcessing): instance of the class DataProcessing.
                Returns:
                    x_train (array): data through which the neural network will be trained.
                    target_train (array): targets of x_train.
                    x_val (array): data through which the neural network is validated.
                    target_val (array): targets of x_val.
            epochs (int): number of iterations of the training cycle (forward propagation + backward propagation). 
            batch_size (int): batch size for training. If batch_size = 1, the neural network is trained using an online learning approach.
                              If batch_size != 1, the neural network is trained using a mini-batch learning approach with batches of size
                              batch_size.
            loss_func (func): loss function.
            d_loss_func (func): derivative of the loss function.
            neural_network (NeuralNetwork): instance of the class NeuralNetwork.
            early_stop (): instance of EarlyStopping Class
        '''
        self.data_splitter = data_splitter
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_func = loss_func
        self.d_loss_func = d_loss_func
        self.early_stop = early_stop
        self.neural_network = neural_network


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
        if self.batch_size > len(num_samples):
            raise ValueError(f'Invalid batch size {self.batch_size}. Must be smaller than number of examples {num_samples}')
        x_batches = []
        target_batches = []
        for i in range(0, num_samples, self.batch_size):
            batch = x_train[i : i + self.batch_size]
            target_batch = target_train[i : i + self.batch_size]
            x_batches.append(batch)
            target_batches.append(target_batch)
        
        return x_batches, target_batches


    def train_epoch(
            self,
            x_train,
            target_train
    ):
        '''
        Function that computes the average training error through the training of the network for a single epoch.
        
        Args:
            x_train (array): one array of the list of arrays of self.data_splitter.x_trains.
            target_train (array): targets corresponding to x_train.
            
        Returns:
            train_error_epoch (float): average training error of one epoch. 
            prediction (array): array of the outputs of the neural network for the training data.
        '''
        batches, target_batches = self.batch_generator(x_train, target_train)
        train_error_epoch = 0
        
        prediction = np.array([])

        for batch, target_batch in zip(batches, target_batches):
            pred = self.neural_network.forward(batch)
            prediction = np.append(prediction, pred)
            train_error_epoch += self.loss_func(target_batch, pred)
            d_loss = self.d_loss_func(target_batch, pred)
            self.neural_network.backward(d_loss)

        train_error_epoch /= x_train.shape[0]

        return train_error_epoch, prediction
    
    
    def train_val(
            self,
            x_val,
            target_val
    ):
        '''
        Function that computes the average validation error through the training of the network for a single epoch.
        
        Args:
            x_val (array): one array of the list of arrays of self.data_splitter.x_vals.
            target_val (array): targets corresponding to x_val.
            
        Returns:
            val_error_epoch (float): average training error of one epoch.
        '''
        pred = self.neural_network.forward(x_val)
        val_error_epoch = self.loss_func(target_val, pred)/x_val.shape[0]

        return val_error_epoch
   


    def train_fold(
            self, 
            stopping_check        
    ):
        '''
        Function that computes training and validation error averaged on the number of folds for each epoch.

        Args:
            stopping_check (Early_Stopping): istances of early stopping class

        Returns:
            train_error_tot (array): training error for each epoch averaged on the number of folds.
            val_error_tot (array): validation error for each epoch averaged on the number of folds.
        
        '''
        train_error_tot = np.zeros(self.epochs)
        val_error_tot = np.zeros(self.epochs)

        iterations = 0
        for x_train, target_train, x_val, target_val in zip(
            self.data_splitter.x_trains,
            self.data_splitter.target_trains,
            self.data_splitter.x_vals,
            self.data_splitter.target_vals
        ):
            iterations += 1
            print(f'\n Begin iteration {iterations} \n')
            train_error = np.array([])
            val_error = np.array([])

            for i in range(self.epochs):
                train_error_epoch, prediction = self.train_epoch(x_train, target_train)
                train_error = np.append(train_error, train_error_epoch)
                val_error_epoch = self.train_val(x_val, target_val)
                val_error = np.append(val_error, val_error_epoch)

                early_check = self.early_stop.stopping_check(i, val_error)
                if early_check:
                    print(f"epoch: {i}")
                    break

                if ((i + 1) % 10 == 0):
                    print(f'epoch {i+1}, train error {train_error_epoch}, val error {val_error_epoch}')

    
            val_error_tot += val_error
            train_error_tot += train_error

            self.neural_network.reinitialize_net_and_optimizers()

        train_error_tot /= self.data_splitter.K
        val_error_tot /= self.data_splitter.K

        return train_error_tot, val_error_tot

'''
Unit test for batches

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

'''

'''
Unit test for the training

for xx, target in zip(data_split.x_trains, data_split.target_trains):
    print(f"\n INIZIO FOLD \n")

    batches, target_batches = train_val.batch_generator(xx, target)
    for i in range(epochs):
        loss = 0
        for batch, target_batch in zip(batches, target_batches):
            pred = nn.forward(batch)
            loss += loss_functions['mse'](target_batch, pred)
            d_loss = d_loss_functions['d_mse'](target_batch, pred)
            
            nn.backward(d_loss)
        loss /= xx.shape[0]
        print(f"loss: \n {loss}")
    
    
    nn.reinitialize_net_and_optimizers()
'''