import numpy as np

from NeuralNetworkClass import NeuralNetwork
from Functions import activation_functions, d_activation_functions, loss_functions, d_loss_functions
from ModelSelectionClass import ModelSelection
from CUPDataProcessing import CUP_data_splitter
from LossControlClass import LossControl

def print_nn_details(nn):
    '''
    Function that prints the Neural Network details.

    Args:
        nn (NeuralNetwork): instance of NeuralNetwork 
    '''
    print("=== Neural Network Details ===")
    
    # Regularization configuration
    print("\nRegularizer Configuration:")
    if hasattr(nn.regularizer, '__dict__'):
        for key, value in nn.regularizer.__dict__.items():
            print(f"  {key}: {value}")
    else:
        print("  No regularizer details available.")
    
    # Layer configuration
    print("\nLayers Configuration:")
    for i, layer in enumerate(nn.layers):
        print(f"    Layer {i + 1}:")
        print(f"    dim_prev_layer: {layer.dim_prev_layer}")
        print(f"    dim_layer: {layer.dim_layer}")
        print(f"    activation_function: {layer.activation_function.__name__}") 
        print(f"    d_activation_function: {layer.d_activation_function.__name__}") 
    
    # Optimization configuration
    print("\nOptimizers Configuration:")
    allowed_optimizer_keys = ['opt_type', 'learning_rate', 'momentum', 'beta_1', 'beta_2', 'epsilon']
    for i, optimizer in enumerate(nn.optimizers):
        print(f"  Optimizer {i + 1}:")
        if hasattr(optimizer, '__dict__'):
            for key, value in optimizer.__dict__.items():
                if key in allowed_optimizer_keys:
                    print(f"    {key}: {value}")
        else:
            print(f"    Optimizer {i + 1} details not available.")


'''
Unit test for NN using CUP data
We use this code to implement phase 1 of our grid search. We explore the grid of hyperparameters
hierarchically, to get a gist of which hyperparameters work and which don't. We then eliminate the parts of
the hyperparameter space in which the hyperparameters don't work, either together or at all.
'''

np.random.seed(12)

# Layer configuration
layers_config = [
    (12, 256, activation_functions['leaky_ReLU'], d_activation_functions['d_leaky_ReLU']),
    (256, 3, activation_functions['linear'], d_activation_functions['d_linear'])
]

# Regulizer configuration
reg_config = {
    'Lambda': 1e-5,
    'alpha' : 0.5,
    'reg_type': 'elastic'
}

# Optimizater configuration
opt_config = {
    'opt_type': 'adam',
    'learning_rate': 3e-5,
    'momentum': 0.9,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'epsilon': 1e-8,
}

# Instance of NeuralNetworkClass
nn = NeuralNetwork(layers_config, reg_config, opt_config)
epochs = 100
batch_size = 1

# Instance of LossControlClass
loss_control = LossControl(epochs)

# Model
train_val = ModelSelection(CUP_data_splitter, epochs, batch_size, loss_functions['mee'], d_loss_functions['d_mee'], nn, loss_control)
train_error_tot, val_error_tot, train_variance, val_variance, smoothness = train_val.train_fold(True, True, True)
#train_error_tot, val_error_tot = train_val.train_fold()

# print(f'errore training \n{train_error_tot}')
# print(f'errore validation \n{val_error_tot}')

print_nn_details(nn)
print(f'smoothness: {smoothness}')
print(f'errore training {train_error_tot[-1]} +- {train_variance[-1]}')
print(f'errore validation {val_error_tot[-1]} +- {val_variance[-1]}')


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

# Neural network characteristics as a multi-line string
legend_info = "\n".join([f"{param}: {value}" for param, value in network_details])

line_train, = plt.plot(train_error_tot, label='Training Error')
line_val, = plt.plot(val_error_tot, label='Validation Error')

plt.xlabel('Epochs', fontsize = 16, fontweight = 'bold')
plt.ylabel('Error', fontsize = 16, fontweight = 'bold')
plt.yscale('log')
plt.grid()
plt.legend(handles = [line_train, line_val], labels = ['Training Error', 'Validation Error'], fontsize = 18, loc = 'best')


# Characteristic are put in a box
props = dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.9)  # Impostazioni della casella
plt.text(
    0.45, 0.95, legend_info, transform=plt.gca().transAxes, fontsize=16,
    verticalalignment='top', horizontalalignment='left', bbox=props
)

# Subplot padding
plt.tight_layout()

plt.tick_params(axis = 'x', labelsize = 16)  # Dimensione xticks
plt.tick_params(axis = 'y', labelsize = 16)  # Dimensione yticks

manager = plt.get_current_fig_manager()
manager.full_screen_toggle() 

plt.pause(2)

# Saving the graph with high resolution
#plt.savefig('grafici/best_config_adam_fine_hl1_5_mee.pdf', bbox_inches = 'tight', dpi = 1200)

plt.show()