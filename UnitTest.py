import numpy as np

#from DataProcessingClass import DataProcessing
from NeuralNetworkClass import NeuralNetwork
from Functions import activation_functions, d_activation_functions, loss_functions, d_loss_functions
from TrainingValidationClass import TrainingValidation
from DataProcessingClass import DataProcessing

'''Unit test for NN'''
np.random.seed(12)

# Configurazione dei layer: [(input_dim, output_dim, activation_function, d_activation_function), ...]
layers_config = [
    (12, 32, activation_functions['ELU'], d_activation_functions['d_ELU']),
    (32, 16, activation_functions['ELU'], d_activation_functions['d_ELU']),
    (16, 3, activation_functions['linear'], d_activation_functions['d_linear'])
]

# Configurazione della regolarizzazione
reg_config = {
    'Lambda': 1e-3,
    'alpha' : 0.5,
    'reg_type': 'elastic'
}

# Configurazione dell'ottimizzazione
opt_config = {
    'opt_type': 'NAG',
    'learning_rate': 1e-6,
    'momentum': 0.9,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'epsilon': 1e-8,
}

nn = NeuralNetwork(layers_config, reg_config, opt_config)

epochs = 500
batch_size = 10

# unit test random 
x_tot = np.random.rand(255, 12)
target_tot = np.random.rand(255, 3)

K = 6
data_split = DataProcessing(x_tot, target_tot, 0.2, K)

train_val = TrainingValidation(data_split, epochs, batch_size, loss_functions['mse'], d_loss_functions['d_mse'], nn)
train_error_tot, val_error_tot = train_val.train_fold()

# Plot degli errori
import matplotlib.pyplot as plt

plt.plot(train_error_tot, label='Training Error')
plt.plot(val_error_tot, label='Validation Error')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.yscale('log')
plt.legend()
plt.savefig('Figure.png')
plt.show()