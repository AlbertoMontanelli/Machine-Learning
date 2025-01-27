import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter

from NeuralNetworkClass import NeuralNetwork
from Functions import activation_functions, d_activation_functions, loss_functions, d_loss_functions
from ModelAssessmentClass import ModelAssessment
from CUPDataProcessing import CUP_data_splitter
from LossControlClass import LossControl

'''
Model assessment for the best configuration of hyperparameters for CUP. 
'''

np.random.seed(12)

#################################################################################################################################

# adam best configuration

#################################################################################################################################

print('Best adam configuration')

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
    'learning_rate': 1e-5,
    'momentum': 0.9,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'epsilon': 1e-8,
}

epochs = 800
batch_size = 1


#################################################################################################################################

# NAG best configuration

#################################################################################################################################

# print('Best NAG configuration')

# # Layer configuration
# layers_config = [
#     (12, 32, activation_functions['leaky_ReLU'], d_activation_functions['d_leaky_ReLU']),
#     (32, 3, activation_functions['linear'], d_activation_functions['d_linear'])
# ]

# # Regulizer configuration
# reg_config = {
#     'Lambda': 1e-4,
#     'alpha' : 0.25,
#     'reg_type': 'elastic'
# }

# # Optimizater configuration
# opt_config = {
#     'opt_type': 'NAG',
#     'learning_rate': 1e-3,
#     'momentum': 0.9,
#     'beta_1': 0.9,
#     'beta_2': 0.999,
#     'epsilon': 1e-8,
# }

# epochs = 11000
# batch_size = 32



# Instance of LossControlClass
loss_control = LossControl(epochs)


# Instance of NeuralNetworkClass
nn = NeuralNetwork(layers_config, reg_config, opt_config)

# Model
assessment = ModelAssessment(CUP_data_splitter.x_train_val,
                            CUP_data_splitter.target_train_val, 
                            CUP_data_splitter.x_test, 
                            CUP_data_splitter.target_test,
                            epochs,
                            batch_size,
                            loss_functions['mee'],
                            d_loss_functions['d_mee'],
                            nn,
                            loss_control
                            )

retrain_error_tot, test_error_tot = assessment.retrain_test(False, False, False)

print('\n')
print('\n')
print(f'retrain error: {retrain_error_tot[-1]}')
print(f'test error: {test_error_tot[-1]}')

network_details = [
    ('Number of Hidden Layers', f'{len(layers_config)-1}'),
    ('Units per Layer', f'{layers_config[0][1]}'),
    ('Activation function', 'Leaky ReLU'),
    ('Batch-size',f"{batch_size}"),
    ('Loss function', 'MEE'),
    ('Learning Rate', f"{opt_config['learning_rate']}"),
    ('Regularization', f"{reg_config['reg_type']}"),
    (r'$\alpha$', f"{reg_config['alpha']}"),
    (r'$\lambda$', f"{reg_config['Lambda']}"),
    ('Optimizer',f"{opt_config['opt_type']}"),
    # (r'$\beta_1$',f"{opt_config['beta_1']}"),
    # (r'$\beta_2$',f"{opt_config['beta_2']}"),
    # (r'$\epsilon$',f"{opt_config['epsilon']}"),
    ('Momentum', f'{opt_config['momentum']}')
]

# Neural network characteristics as a multi-line string
legend_info = "\n".join([f"{param}: {value}" for param, value in network_details])

line_train, = plt.plot(retrain_error_tot, label='Training Error')
line_val, = plt.plot(test_error_tot, label='Test Error')

plt.xlabel('Epochs', fontsize = 16, fontweight = 'bold')
plt.ylabel('MEE Error', fontsize = 16, fontweight = 'bold')
plt.yscale('log')
plt.grid()
plt.legend(handles = [line_train, line_val], labels = ['Training Error', 'Test Error'], fontsize = 25, loc = 'best')

ax = plt.gca()  
ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
ax.yaxis.set_major_formatter(ScalarFormatter())

# Characteristics are put in a box
plt.text(0.45, 0.970, "Network characteristics", transform = ax.transAxes, fontsize = 20, 
        fontweight = 'bold', horizontalalignment = 'center', verticalalignment = 'top')

props = dict(boxstyle = 'round', facecolor = 'white', edgecolor = 'black')
plt.text(0.45, 0.930, legend_info, transform = ax.transAxes, fontsize = 20, 
        verticalalignment = 'top', horizontalalignment = 'center', bbox = props)

# Subplot padding
plt.tight_layout()

plt.tick_params(axis = 'x', labelsize = 16)  # Xticks dimension
plt.tick_params(axis = 'y', labelsize = 16)  # Yticks dimension

plt.title('Training error vs Test error', fontsize = 20, fontweight = 'bold')

manager = plt.get_current_fig_manager()
manager.full_screen_toggle() 

plt.pause(2)

# Saving the graph with high resolution
plt.savefig(f'grafici_per_slides/adam_final_best_config.pdf', bbox_inches = 'tight', dpi = 1200)

plt.show()


