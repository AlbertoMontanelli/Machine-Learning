import numpy as np
from DataProcessingClass import DataProcessing
from NeuralNetworkClass import NeuralNetwork
from Functions import activation_functions, d_activation_functions, loss_functions, d_loss_functions

class TrainingValidation:

    def __init__(
            self,
            data_splitter,
            epochs,
            batch_size,
            loss_func,
            d_loss_func,
            neural_network
    ):
        '''
        Class focused on the actual training and validation of the neural network.

        Args:
            data_splitter (DataProcessing): instance of the class DataProcessing.
            epochs (int): number of iterations of the cycle forward propagation + backward propagation + weights update. 
            batch_size (int): batch size for training. If batch_size = 1, the neural network is trained using an online learning approach.
                              If batch_size != 1, the neural network is trained using a mini-batch learning approach with batches of size
                              batch_size.
            loss_func (func): loss function.
            d_loss_func (func): derivative of loss function.
            neural_network (NeuralNetwork): instance of the class NeuralNetwork.
        '''
        self.data_splitter = data_splitter
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_func = loss_func
        self.d_loss_func = d_loss_func
        self.neural_network = neural_network


    def batch_generator( # eventualmente metterlo nell'init
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
        '''
        batches, target_batches = self.batch_generator(x_train, target_train)
        train_error_epoch = 0

        for batch, target_batch in zip(batches, target_batches):
            pred = self.neural_network.forward(batch)
            train_error_epoch += self.loss_func(target_batch, pred)
            d_loss = self.d_loss_func(target_batch, pred)
            self.neural_network.backward(d_loss)

        train_error_epoch /= x_train.shape[0]

        return train_error_epoch
    
    
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
            self            
    ):
        '''
        Function that computes training and validation error averaged on the number of folds for each epoch

        Returns:
            train_error_tot (array): Training error array averaged on the number of folds for each epoch.
            val_error_tot (array): Validation error array averaged on the number of folds for each epoch.
        
        '''
        train_error_tot = np.zeros(epochs)
        val_error_tot = np.zeros(epochs)
        a=0
        for x_train, target_train, x_val, target_val in zip(
            self.data_splitter.x_trains,
            self.data_splitter.target_trains,
            self.data_splitter.x_vals,
            self.data_splitter.target_vals
        ):
            a += 1
            print(f'\n Inizio iterazion {a} \n')
            train_error = np.array([])
            val_error = np.array([])

            for i in range(self.epochs):
                train_error_epoch = self.train_epoch(x_train, target_train)
                train_error = np.append(train_error, train_error_epoch)

                val_error_epoch = self.train_val(x_val, target_val)
                val_error = np.append(val_error, val_error_epoch)
                print(f'epoch {i+1}, train error {train_error_epoch}, val error {val_error_epoch}')

            val_error_tot += val_error
            train_error_tot += train_error
            self.neural_network.reinitialize_net_and_optimizers()

        train_error_tot /= self.data_splitter.K
        val_error_tot /= self.data_splitter.K

        return train_error_tot, val_error_tot


'''Unit test for NN'''
np.random.seed(42)

# Configurazione dei layer: [(input_dim, output_dim, activation_function, d_activation_function), ...]
layers_config = [
    (15, 10, activation_functions['ELU'], d_activation_functions['d_ELU']),
    (10, 6, activation_functions['ELU'], d_activation_functions['d_ELU']),
    (6, 3, activation_functions['linear'], d_activation_functions['d_linear'])
]

# Configurazione della regolarizzazione
reg_config = {
    'Lambda_t': 0.0001,
    'Lambda_l': 0.0001,
    'alpha': 0.5,
    'reg_type': 'elastic'
}

# Configurazione dell'ottimizzazione
opt_config = {
    'opt_type': 'NAG',
    'learning_rate_w': 0.001,
    'learning_rate_b': 0.001,
    'momentum': 0.9,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'epsilon': 1e-8,
}

nn = NeuralNetwork(layers_config, reg_config, opt_config)

x_tot = np.random.rand(1000, 15)
target_tot = np.random.rand(1000, 3)

K = 5
data_split = DataProcessing(x_tot, target_tot, 0.2, K)

epochs = 100
batch_size = 30

train_val = TrainingValidation(data_split, epochs, batch_size, loss_functions['mse'], d_loss_functions['d_mse'], nn)
train_error_tot, val_error_tot = train_val.train_fold()

print(f'train error: \n {train_error_tot} \n val error: \n {val_error_tot}')

# Plot degli errori
import matplotlib.pyplot as plt

plt.plot(train_error_tot, label='Training Error')
plt.plot(val_error_tot, label='Validation Error')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.yscale('log')
plt.legend()
plt.show()

'''
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

'''Unit test for batches
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