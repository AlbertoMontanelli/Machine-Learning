import numpy as np
from DataProcessingClass import DataProcessing
from NeuralNetworkClass import NeuralNetwork
from Functions import activation_functions, d_activation_functions, loss_functions, d_loss_functions


'''Unit test for NN'''
np.random.seed(42)

# Configurazione dei layer: [(input_dim, output_dim, activation_function, d_activation_function), ...]
layers_config = [
    (5, 4, activation_functions['linear'], d_activation_functions['d_linear']),
    (4, 3, activation_functions['linear'], d_activation_functions['d_linear'])
]

# Configurazione della regolarizzazione
reg_config = {
    'Lambda_t': 0.01,
    'Lambda_l': 0.01,
    'alpha': 1e-4,
    'reg_type': 'elastic'
}

# Configurazione dell'ottimizzazione
opt_config = {
    'opt_type': 'NAG',
    'learning_rate_w': 0.0001,
    'learning_rate_b': 0.0001,
    'momentum': 0.9,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'epsilon': 1e-8,
}


nn = NeuralNetwork(layers_config, reg_config, opt_config)

x_tot = np.random.rand(1000, 5)
target_tot = np.random.rand(1000, 3)

K = 5
data_split = DataProcessing(x_tot, target_tot, 0, K)

epochs = 10
batch_size = 50

#train_val = TrainingValidation(data_split, epochs, batch_size)

def batch_generator(x, target, batch_size):

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

for xx, target in zip(data_split.x_trains, data_split.target_trains):
    print(f"\n INIZIO FOLD \n")

    nn.reinitialize_weights_and_optimizers()

    batches, target_batches = batch_generator(xx, target, batch_size)
    for i in range(epochs):
        loss = 0
        for batch, target_batch in zip(batches, target_batches):
            pred = nn.forward(batch)
            loss += loss_functions['mse'](target_batch, pred)
            d_loss = d_loss_functions['d_mse'](target_batch, pred)
            
            nn.backward(d_loss)
        loss /= xx.shape[0]
        print(f"loss: \n {loss}")
    


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
    

"""
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
"""