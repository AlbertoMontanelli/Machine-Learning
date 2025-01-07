import numpy as np

from Functions import activation_functions, d_activation_functions, loss_functions, d_loss_functions
from DataProcessingClass import DataProcessing
from NeuralNetworkClass import NeuralNetwork
from TrainingValidationClass import TrainValidation

np.random.seed(42)

# Configurazione dei layer: [(input_dim, output_dim, activation_function, d_activation_function), ...]
layers_config = [
    (15, 10, activation_functions['ELU'], d_activation_functions['d_ELU']),
    (10, 6, activation_functions['ELU'], d_activation_functions['d_ELU']),
    (6, 3, activation_functions['linear'], d_activation_functions['d_linear'])
]

# Configurazione della regolarizzazione
reg_config = {
    'Lambda_t': 0.001,
    'Lambda_l': 0.001,
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

x_tot = np.random.rand(1000, 15)
target_tot = np.random.rand(1000, 3)

# Configurazione delle classi
neural_network = NeuralNetwork(layers_config, reg_config, opt_config)
data_split = DataProcessing(x_tot, target_tot, test_perc=0.2, K=5)
train = TrainValidation(neural_network, data_split)

# Funzioni di loss
loss_function = loss_functions['mse']
loss_function_derivative = d_loss_functions['d_mse']

# Esecuzione
train_error, val_error = train.execute(
    epochs=100, 
    batch_size=30, 
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