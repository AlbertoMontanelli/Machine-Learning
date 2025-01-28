import numpy as np
from matplotlib.ticker import LogLocator, ScalarFormatter

from NeuralNetworkClass import NeuralNetwork
from Functions import activation_functions, d_activation_functions, loss_functions, d_loss_functions
from ModelSelectionClass import ModelSelection
from CUPDataProcessing import CUP_data_splitter
from LossControlClass import LossControl


np.random.seed(12)

########################################################################################################################

# TRAINING OF THE NEURAL NETWORK USING MONK3_DATA.

########################################################################################################################

'''
This part of code implements the training and evaluation of a neural network on a dataset called monk3_data. 
The objective is to analyze the network's performance by calculating the training and test errors as well as their accuracy.
After running the script, the following outputs are obtained:
    Training and test error curves, which help assess overfitting or underfitting.
    Training and test accuracy curves, to understand the model's ability to generalize.
'''

# Layer configuration
layers_config = [
    (12, 32, activation_functions['leaky_ReLU'], d_activation_functions['d_leaky_ReLU']),
    (32, 3, activation_functions['linear'], d_activation_functions['d_linear'])
]

# Regularization configuration
reg_config_adam = {
    'Lambda': 1e-6,
    'alpha' : 0.5,
    'reg_type': 'elastic'
}

# Optimization configuration
opt_config_adam = {
    'opt_type': 'adam',
    'learning_rate': 1e-5,
    'momentum': 0.9,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'epsilon': 1e-8,
}

# Regularization configuration
reg_config_none = {
    'Lambda': 1e-6,
    'alpha' : 0.5,
    'reg_type': 'elastic'
}

# Optimization configuration
opt_config_none = {
    'opt_type': 'none',
    'learning_rate': 1e-5,
    'momentum': 0.9,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'epsilon': 1e-8,
}

epochs = 1000
batch_size = 1


# Instance of NeuralNetworkClass
nn_adam = NeuralNetwork(layers_config, reg_config_adam, opt_config_adam)
nn_none = NeuralNetwork(layers_config, reg_config_none, opt_config_none)

# Instance of LossControlClass
loss_control = LossControl(epochs)

# Model
train_val_none = ModelSelection(CUP_data_splitter, epochs, batch_size, loss_functions['mee'], d_loss_functions['d_mee'], nn_none, loss_control)
train_error_none, val_error_none, train_variance_none, val_variance_none = train_val_none.train_fold(False, False, False)

np.random.seed(12)

train_val_adam = ModelSelection(CUP_data_splitter, epochs, batch_size, loss_functions['mee'], d_loss_functions['d_mee'], nn_adam, loss_control)
train_error_adam, val_error_adam, train_variance_adam, val_variance_adam = train_val_adam.train_fold(False, False, False)





#print(f'smoothness: {smoothness}')
print(f'errore training {train_error_adam[-1]} +- {train_variance_adam[-1]}')
print(f'errore validation {val_error_adam[-1]} +- {val_variance_adam[-1]}')
print(f'errore training {train_error_none[-1]} +- {train_variance_none[-1]}')
print(f'errore validation {val_error_none[-1]} +- {val_variance_none[-1]}')


#############################################################################################################################

# PLOT

##############################################################################################################################

import matplotlib.pyplot as plt


network_details = [
    ('Number of Hidden Layers', f'{len(layers_config)-1}'),
    ('Units per Layer', f'{layers_config[0][1]}'),
    ('Activation function', 'Leaky ReLU'),
    ('Batch-size',f"{batch_size}"),
    ('Loss function', 'MEE'),
    ('Learning Rate', f"{opt_config_adam['learning_rate']}"),
    ('Regularization', f"{reg_config_adam['reg_type']}"),
    (r'$\alpha$', f"{reg_config_adam['alpha']}"),
    (r'$\lambda$', f"{reg_config_adam['Lambda']}"),
    # ('For optimizer',f"{opt_config_adam['opt_type']}"),
    # (r'$\beta_1$',f"{opt_config_adam['beta_1']}"),
    # (r'$\beta_2$',f"{opt_config_adam['beta_2']}"),
    # (r'$\epsilon$',f"{opt_config_adam['epsilon']}"),
    # ('For optimizer',f"{opt_config_none['opt_type']}"),
    # ('Regularization', f"{reg_config_none['reg_type']}"),
    # ('Momentum', f'{opt_config['momentum']}')
]

# Neural network characteristics as a multi-line string
legend_info = "\n".join([f"{param}: {value}" for param, value in network_details])

plt.plot(train_error_adam, label = 'Training Error Adam', linewidth = 2, color = 'cornflowerblue', linestyle = '--')
plt.plot(val_error_adam, label = 'Test Error Adam', linewidth = 2, color = 'sandybrown', linestyle = '--')
plt.plot(train_error_none, label = 'Training Error no opt', linewidth = 2, color = 'cornflowerblue', linestyle = '-')
plt.plot(val_error_none, label = 'Test Error no opt', linewidth = 2, color = 'sandybrown', linestyle = '-')

plt.xlabel('Epochs', fontsize = 16, fontweight = 'bold')
plt.ylabel('MEE Error', fontsize = 16, fontweight = 'bold')
plt.yscale('log')
plt.grid()
plt.legend(labels = ['Training error adam', 'Validation error adam', 'Training error no opt', 'Validation error no opt'], fontsize = 20, loc = 'best')

ax = plt.gca()  
ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
ax.yaxis.set_major_formatter(ScalarFormatter())


# Characteristics are put in a box
plt.text(0.4, 0.970, "Network characteristics", transform = ax.transAxes, fontsize = 18, 
        fontweight = 'bold', horizontalalignment = 'center', verticalalignment = 'top')

props = dict(boxstyle = 'round', facecolor = 'white', edgecolor = 'black')
plt.text(0.4, 0.930, legend_info, transform = ax.transAxes, fontsize = 18, 
        verticalalignment = 'top', horizontalalignment = 'center', bbox = props)

# Subplot padding
plt.tight_layout()

plt.tick_params(axis = 'x', labelsize = 16)  # Dimensione xticks
plt.tick_params(axis = 'y', labelsize = 16)  # Dimensione yticks

plt.title('Training error and Validation error for adam, no opt.', fontsize = 20, fontweight = 'bold')

manager = plt.get_current_fig_manager()
manager.full_screen_toggle() 

plt.pause(2)

# Saving the graph with high resolution
plt.savefig('grafici_per_slides/CUP_adam_vs_none.pdf', bbox_inches = 'tight', dpi = 1200)

plt.show()