import numpy as np
import pandas as pd

from DataProcessingClass import DataProcessing
from NeuralNetworkClass import NeuralNetwork
from Functions import activation_functions, d_activation_functions, loss_functions, d_loss_functions
from TrainingValidationClass import TrainingValidation

from MonkDataProcessing import training_set_1, target_training_set_1, test_set_1, target_test_set_1

# Splitting of training set and validation set
data_splitter_monk1_selection = DataProcessing(training_set_1, target_training_set_1, test_perc = 0., K = 5)

# Training of the neural network using monk1_data

# Layers configuration
layers_config = [
    (17, 8, activation_functions['leaky_ReLU'], d_activation_functions['d_leaky_ReLU']),
    (8, 1, activation_functions['sigmoid'], d_activation_functions['d_sigmoid'])
]

# Regularization configuration
reg_config = {
    'Lambda': 0.0001,
    'alpha' : 0.5,
    'reg_type': 'elastic'
}

# Optimization configuration
opt_config = {
    'opt_type': 'NAG',
    'learning_rate': 0.00001,
    'momentum': 0.9,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'epsilon': 1e-8,
}

nn_selection = NeuralNetwork(layers_config, reg_config, opt_config)

epochs = 500
batch_size = 22

train_val = TrainingValidation(data_splitter_monk1_selection, epochs, batch_size, loss_functions['bce'], d_loss_functions['d_bce'], nn_selection)
train_error_tot, val_error_tot = train_val.train_fold()

# Plot
import matplotlib.pyplot as plt

# Modifica del font della label nella legenda
font = {'family': 'serif', 'weight': 'normal', 'size': 24}


plt.plot(train_error_tot, label = 'Training Error')
plt.plot(val_error_tot, label = 'Validation Error')
plt.xlabel('Epochs', fontdict = font)
plt.ylabel('Error', fontdict = font)
plt.yscale('log')
plt.grid()
plt.legend()
plt.show()

# Model assessment with accuracy plot

nn_assessment = NeuralNetwork(layers_config, reg_config, opt_config)

train_val = TrainingValidation(data_splitter_monk1_assessment, epochs, batch_size, loss_functions['bce'], d_loss_functions['d_bce'], nn_selection)
retrain_error_tot, zero_val = train_val.train_fold()
