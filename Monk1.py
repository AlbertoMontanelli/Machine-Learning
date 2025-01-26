import numpy as np

from NeuralNetworkClass import NeuralNetwork
from Functions import activation_functions, d_activation_functions, loss_functions, d_loss_functions
from ModelAssessmentClass import ModelAssessment
from LossControlClass import LossControl

from MonkDataProcessing import monk_data

########################################################################################################################

# TRAINING OF THE NEURAL NETWORK USING MONK1_DATA.

########################################################################################################################

'''
This part of code implements the training and evaluation of a neural network on a dataset called monk1_data. 
The objective is to analyze the network's performance by calculating the training and test errors as well as their accuracy.
After running the script, the following outputs are obtained:
    Training and test error curves, which help assess overfitting or underfitting.
    Training and test accuracy curves, to understand the model's ability to generalize.
'''

np.random.seed(12)

# Layers configuration
layers_config = [
    (17, 4, activation_functions['sigmoid'], d_activation_functions['d_sigmoid']),
    (4, 1, activation_functions['sigmoid'], d_activation_functions['d_sigmoid'])
]

# Regularization configuration
reg_config = {
    'Lambda': 0,
    'alpha' : 0,
    'reg_type': 'none'
}

# Optimization configuration
opt_config = {
    'opt_type': 'none',
    'learning_rate': 0.3,
    'momentum': 0.9,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'epsilon': 1e-8,
}

epochs = 1000
batch_size = 1


########################################################################################################################

# MODEL ASSESSMENT FOR MONK1 DATASET WITH TRAINING, TEST ERROR CURVE AND TRAINING, TEST ACCURACY CURVE

########################################################################################################################

nn_assessment = NeuralNetwork(layers_config, reg_config, opt_config)
loss_control = LossControl(epochs)

train_test = ModelAssessment(
    monk_data['training_set_1'], 
    monk_data['target_training_set_1'], 
    monk_data['test_set_1'], 
    monk_data['target_test_set_1'], 
    epochs, 
    batch_size, 
    loss_functions['mse'], 
    d_loss_functions['d_mse'], 
    nn_assessment,
    loss_control,
    classification_problem = True
    )

train_error, test_error, accuracy_train, accuracy_test = train_test.retrain_test()


########################################################################################################################

# PLOT

########################################################################################################################

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext

# Figure
fig, ax = plt.subplots(1, 2, figsize = (15, 10))

# Loss functions graph
line_train, = ax[0].plot(train_error, label = 'Training Error', linewidth = 2)
line_test, = ax[0].plot(test_error, label = 'Test Error', linewidth = 2)
ax[0].set_xlabel('Epochs', fontsize = 16, fontweight = 'bold')
ax[0].set_ylabel('Error', fontsize = 16, fontweight = 'bold')
ax[0].set_yscale('log')
ax[0].yaxis.set_major_locator(LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
ax[0].yaxis.set_major_formatter(LogFormatterMathtext(base=10)) # Configure the y-axis to display labels in scientific notation.
ax[0].grid()

# Accuracy graph
line_train_acc, = ax[1].plot(accuracy_train, label = 'Training accuracy', linewidth = 2)
line_test_acc, = ax[1].plot(accuracy_test, label = 'Test accuracy', linewidth = 2)
ax[1].set_xlabel('Epochs', fontsize = 16, fontweight = 'bold')
ax[1].set_ylabel('Accuracy', fontsize = 16, fontweight = 'bold')
ax[1].grid()

# Creation of a legend containing the details of the neural network
network_details = [
    ('Number of Hidden Layers', f'{layers_config[1][1]}'),
    ('Units per Layer', f'{layers_config[1][0]}'),
    ('Activation function', 'sigmoid'),
    ('Loss function', 'mse'),
    ('Learning Rate', f"{opt_config['learning_rate']}"),
    ('Regularization', f"{reg_config['reg_type']}"),
    #('lambda', f"{reg_config['Lambda']}"),
    #('alpha', f"{reg_config['alpha']}"),
    ('Optimizer',f"{opt_config['opt_type']}")
]

legend_info = "\n".join([f"{param}: {value}" for param, value in network_details])

ax[0].legend(handles = [line_train, line_test], labels = ['Training Error', 'Test Error'], fontsize = 18)
ax[1].legend(handles = [line_train_acc, line_test_acc], labels = ['Training Accuracy', 'Test Accuracy'], 
             fontsize = 18, loc = 'best')

ax[1].text(0.72, 0.305, "Network characteristics", transform = ax[1].transAxes, fontsize = 16, 
        fontweight = 'bold', horizontalalignment = 'center', verticalalignment = 'top')

props = dict(boxstyle = 'round', facecolor = 'white', edgecolor = 'black')
ax[1].text(0.72, 0.27, legend_info, transform = ax[1].transAxes, fontsize = 16, 
        verticalalignment = 'top', horizontalalignment = 'center', bbox = props)

ax[0].set_title('Error vs Epochs', fontsize = 18, fontweight = 'bold')
ax[1].set_title('Accuracy vs Epochs', fontsize = 18, fontweight = 'bold')

plt.tight_layout()

for a in ax:
    a.tick_params(axis = 'x', labelsize = 16)
    a.tick_params(axis = 'y', labelsize = 16)

manager = plt.get_current_fig_manager()
manager.full_screen_toggle() 

plt.pause(2)

plt.savefig(f"grafici/monk1_batch{batch_size}_learningrate{opt_config['learning_rate']}.pdf", bbox_inches = 'tight', dpi = 1200)

plt.show()