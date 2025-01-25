import numpy as np
import matplotlib.pyplot as plt

from NeuralNetworkClass import NeuralNetwork
from Functions import activation_functions, d_activation_functions, loss_functions, d_loss_functions
from ModelAssessmentClass import ModelAssessment
from CUPDataProcessing import CUP_data_splitter
from LossControlClass import LossControl

'''
Model assessment for the best configuration of hyperparameters for CUP. 
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
    'learning_rate': 1e-5,
    'momentum': 0.9,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'epsilon': 1e-8,
}

epochs = 800
batch_size = 1

# Instance of LossControlClass
loss_control = LossControl(epochs)

nn_total = []

retrain_error_avg = []
test_error_avg = []

for i in range(5):
    print(f'Model Assessment n {i+1}')
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
    retrain_error_avg.append(retrain_error_tot)
    test_error_avg.append(test_error_tot)


retrain_error = (np.sum(retrain_error_avg, axis=0))/5
test_error = (np.sum(test_error_avg, axis=0))/5
retrain_variance = np.std(retrain_error_avg, axis = 0, ddof = 1)
test_variance = np.std(test_error_avg, axis = 0, ddof = 1)

print('\n')
print('\n')
print(f'retrain error: {retrain_error[-1]} +- {retrain_variance[-1]}')
print(f'test error: {test_error[-1]} +- {test_variance[-1]}')

network_details = [
    ('Number of Hidden Layers', f'{len(layers_config)}'),
    ('Units per Layer', f'{layers_config[0][1]}'),
    ('Activation function', 'Leaky ReLU'),
    ('Loss function', 'mee'),
    ('Learning Rate', f"{opt_config['learning_rate']}"),
    ('Regularization', f"{reg_config['reg_type']}"),
    ('Lambda', f"{reg_config['Lambda']}"),
    ('Optimizer',f"{opt_config['opt_type']}"),
    ('Batch-size',f"{batch_size}")
]

# Neural network characteristics as a multi-line string
legend_info = "\n".join([f"{param}: {value}" for param, value in network_details])

line_train, = plt.plot(retrain_error, label='Retraining Error')
line_val, = plt.plot(test_error, label='Test Error')

plt.xlabel('Epochs', fontsize = 16, fontweight = 'bold')
plt.ylabel('Error', fontsize = 16, fontweight = 'bold')
plt.yscale('log')
plt.grid()
plt.legend(handles = [line_train, line_val], labels = ['Retraining Error', 'Test Error'], fontsize = 18, loc = 'best')


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
#plt.savefig(f'grafici/adam_best_config_seed_{i}.pdf', bbox_inches = 'tight', dpi = 1200)

plt.show()


