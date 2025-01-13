import numpy as np
import pandas as pd

from DataProcessingClass import DataProcessing
from NeuralNetworkClass import NeuralNetwork
from Functions import activation_functions, d_activation_functions, loss_functions, d_loss_functions
from TrainingValidationClass import TrainingValidation

# Importing data
file_path_train = "monk+s+problems/monks-1.train"
file_path_test = "monk+s+problems/monks-1.test"
columns = ["target", "feat1", "feat2", "feat3", "feat4", "feat5", "feat6", "Id"]  
df_train = pd.read_csv(file_path_train, sep = " ", header = None, names = columns, skipinitialspace = True)
df_test = pd.read_csv(file_path_test, sep = " ", header = None, names = columns, skipinitialspace = True)
training_set1 = df_train.drop(columns = ["target", "Id"])
test_set1 = df_test.drop(columns = ["target", "Id"])

# One-hot encoding of train_val_set and test_set for data and targets
training_set1 = pd.get_dummies(training_set1, columns = training_set1.columns[1:], drop_first = False, dtype = int)
test_set1 = pd.get_dummies(test_set1, columns = test_set1.columns[1:], drop_first = False, dtype = int)
target_train_set1 = df_train.drop(columns = ["feat1", "feat2", "feat3", "feat4", "feat5", "feat6", "Id"])
target_test_set1 = df_test.drop(columns = ["feat1", "feat2", "feat3", "feat4", "feat5", "feat6", "Id"])

# Splitting of training set and validation set
monk1_data = DataProcessing(training_set1, target_train_set1, test_perc = 0., K = 5)

# Training of the neural network using monk1_data
# Configurazione dei layer: [(input_dim, output_dim, activation_function, d_activation_function), ...]
layers_config = [
    (17, 8, activation_functions['sigmoid'], d_activation_functions['d_sigmoid ']),
    (8, 3, activation_functions['sigmoid'], d_activation_functions['d_sigmoid'])
]

# Configurazione della regolarizzazione
reg_config = {
    'Lambda': 0.0001,
    'alpha' : 0.5,
    'reg_type': 'elastic'
}

# Configurazione dell'ottimizzazione
opt_config = {
    'opt_type': 'NAG',
    'learning_rate': 0.00001,
    'momentum': 0.9,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'epsilon': 1e-8,
}

nn = NeuralNetwork(layers_config, reg_config, opt_config)

epochs = 100
batch_size = 20

train_val = TrainingValidation(monk1_data, epochs, batch_size, loss_functions['bce'], d_loss_functions['d_bce'], nn)
train_error_tot, val_error_tot = train_val.train_fold()

# Plot degli errori
import matplotlib.pyplot as plt

plt.plot(train_error_tot, label='Training Error')
plt.plot(val_error_tot, label='Validation Error')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.yscale('log')
plt.legend()
plt.show()

# Plot accuracy
