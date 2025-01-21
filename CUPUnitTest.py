import numpy as np

# from DataProcessingClass import DataProcessing
from NeuralNetworkClass import NeuralNetwork
from Functions import activation_functions, d_activation_functions, loss_functions, d_loss_functions
from ModelSelectionClass import ModelSelection
from CUPDataProcessing import CUP_data_splitter
from EarlyStoppingClass import EarlyStopping

'''Unit test for NN'''
np.random.seed(12)

# Layer configuration: [(input_dim, output_dim, activation_function, d_activation_function), ...]
layers_config = [
    (12, 128, activation_functions['leaky_ReLU'], d_activation_functions['d_leaky_ReLU']),
    (128, 3, activation_functions['linear'], d_activation_functions['d_linear'])
]

# Regulizer configuration
reg_config = {
    'Lambda': 1e-3,
    'alpha' : 0.33,
    'reg_type': 'elastic'
}

# Optimizater configuration
opt_config = {
    'opt_type': 'adam',
    'learning_rate': 1e-4,
    'momentum': 0.9,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'epsilon': 1e-8,
}

nn = NeuralNetwork(layers_config, reg_config, opt_config)

epochs = 500
batch_size = 25 # len(CUP_data_splitter.x_trains[0])
print(f'batchsize: {batch_size}')
early_stop = EarlyStopping(epochs)

train_val = ModelSelection(CUP_data_splitter, epochs, batch_size, loss_functions['mse'], d_loss_functions['d_mse'], nn, early_stop)
train_error_tot, val_error_tot = train_val.train_fold(True)

#############################################################################################################################

# PLOT

##############################################################################################################################

import matplotlib.pyplot as plt


network_details = [
    ('Number of Hidden Layers', f'{len(layers_config)}'),
    ('Units per Layer', f'{layers_config[0][1]}'),
    ('Activation function', 'sigmoid'),
    ('Loss function', 'Leaky ReLU'),
    ('Learning Rate', f"{opt_config['learning_rate']}"),
    ('Regularization', f"{reg_config['reg_type']}"),
    ('Lambda', f"{reg_config['Lambda']}"),
    ('Optimizer',f"{opt_config['opt_type']}"),
    ('Batch-size',f"{batch_size}")
]

# Aggiungere informazioni della rete come stringa multilinea
legend_info = "\n".join([f"{param}: {value}" for param, value in network_details])

line_train, = plt.plot(train_error_tot, label='Training Error')
line_val, = plt.plot(val_error_tot, label='Validation Error')

plt.xlabel('Epochs', fontsize = 16, fontweight = 'bold')
plt.ylabel('Error', fontsize = 16, fontweight = 'bold')
plt.yscale('log')
plt.grid()
plt.legend(handles = [line_train, line_val], labels = ['Training Error', 'Validation Error'], fontsize = 18, loc = 'best')


# Aggiungere un riquadro con informazioni della rete
props = dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.9)  # Impostazioni della casella
plt.text(
    0.45, 0.95, legend_info, transform=plt.gca().transAxes, fontsize=16,
    verticalalignment='top', horizontalalignment='left', bbox=props
)

# Aggiungere padding tra i subplot
plt.tight_layout()

plt.tick_params(axis = 'x', labelsize = 16)  # Dimensione xticks
plt.tick_params(axis = 'y', labelsize = 16)  # Dimensione yticks

# Mettere il grafico a schermo intero
manager = plt.get_current_fig_manager()
manager.full_screen_toggle() 

plt.pause(2)  # Pausa di 2 secondi

# Salvare il grafico in PDF con alta risoluzione
plt.savefig('grafici/70.pdf', bbox_inches = 'tight', dpi = 1200)

plt.show()