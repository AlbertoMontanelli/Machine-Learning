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

        Yields:
            (array, array): batch of input data and labels.
        '''
        indices = np.arange(x.shape[0])
        #np.random.shuffle(indices)  # VA SHUFFOLATO ANCHE QUESTO?
        for start in range(0, x.shape[0], batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            yield x[batch_indices], target[batch_indices]

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
        for x_batch, target_batch in self.batch_generator(x_train, target_train, batch_size):
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


    def execute(self, epochs, batch_size, loss_function, loss_function_derivative): # DA RICONTROLLARE SOPRATTUTTO CAPIRE
        # SE FA LA VAAIDARION PER OGNI FOLD E CHE SHAPE HA L'ARRAY results 
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
            self.neural_network.reinitialize_weights()  # Reinitialize weights for each fold

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


## UNIT TEST

np.random.seed(42)

# Configurazione dei layer: [(input_dim, output_dim, activation_function, d_activation_function), ...]
layers_config = [
    (15, 12, activation_functions['sigmoid'], d_activation_functions['d_sigmoid']),
    (12, 10, activation_functions['tanh'], d_activation_functions['d_tanh']),
    (10, 3, activation_functions['ReLU'], d_activation_functions['d_ReLU'])
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
    'learning_rate_w': 0.001,
    'learning_rate_b': 0.001,
    'momentum': 0.9,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'epsilon': 1e-8,
}

x_tot = np.random.rand(10, 15)
target_tot = np.random.rand(10, 3)

# Configurazione delle classi
neural_network = NeuralNetwork(layers_config, reg_config, opt_config)
data_split = DataProcessing(x_tot, target_tot, test_perc=0.2, K=5)
train = TrainValidation(neural_network, data_split)

# Funzioni di loss
loss_function = loss_functions['mse']
loss_function_derivative = d_loss_functions['d_mse']

# Esecuzione
train_error, val_error = train.execute(
    epochs=50, 
    batch_size=32, 
    loss_function=loss_function, 
    loss_function_derivative=loss_function_derivative
)

# Plot degli errori
import matplotlib.pyplot as plt

plt.plot(train_error, label='Training Error')
plt.plot(val_error, label='Validation Error')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.yscale('log')
plt.legend()
plt.show()