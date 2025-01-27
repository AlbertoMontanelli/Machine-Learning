import numpy as np

from NeuralNetworkClass import NeuralNetwork
from Functions import activation_functions, d_activation_functions, loss_functions, d_loss_functions
from ModelAssessmentClass import ModelAssessment
from LossControlClass import LossControl

from MonkDataProcessing import monk_data

np.random.seed(12)

########################################################################################################################

# TRAINING OF THE NEURAL NETWORK USING MONK2_DATA.

########################################################################################################################

'''
This part of code implements the training and evaluation of a neural network on a dataset called monk2_data. 
The objective is to analyze the network's performance by calculating the training and test errors as well as their accuracy.
After running the script, the following outputs are obtained:
    Training and test error curves, which help assess overfitting or underfitting.
    Training and test accuracy curves, to understand the model's ability to generalize.
'''

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

epochs = 500
batch_size = 1


########################################################################################################################

# MODEL ASSESSMENT FOR MONK2 DATASET WITH TRAINING, TEST ERROR CURVE AND TRAINING, TEST ACCURACY CURVE

########################################################################################################################

nn_assessment = NeuralNetwork(layers_config, reg_config, opt_config)
loss_control = LossControl(epochs)

train_test = ModelAssessment(
    monk_data['training_set_2'], 
    monk_data['target_training_set_2'], 
    monk_data['test_set_2'], 
    monk_data['target_test_set_2'], 
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
plt.figure()

# Loss functions graph
plt.plot(train_error, label = 'Training Error', linewidth = 2)
plt.plot(test_error, label = 'Test Error', linewidth = 2)
plt.xlabel('Epochs', fontsize = 18, fontweight = 'bold')
plt.ylabel('MSE Error', fontsize = 18, fontweight = 'bold')
plt.yscale('log')

ax = plt.gca()  # Ottieni gli assi correnti
ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))

plt.grid()



# Creation of a legend containing the details of the neural network
network_details = [
    ('Number of Hidden Layers', f'{layers_config[1][1]}'),
    ('Units per Layer', f'{layers_config[1][0]}'),
    ('Activation function', 'sigmoid'),
    ('Loss function', 'MSE'),
    ('Learning Rate', f"{opt_config['learning_rate']}"),
    ('Regularization', f"{reg_config['reg_type']}"),
    #('lambda', f"{reg_config['Lambda']}"),
    #('alpha', f"{reg_config['alpha']}"),
    ('Optimizer',f"{opt_config['opt_type']}"),
    ('Batch size', f"{batch_size}")
]

legend_info = "\n".join([f"{param}: {value}" for param, value in network_details])

plt.legend(labels = ['Training Error', 'Test Error'], fontsize = 25, loc = 'lower left')


plt.text(0.86, 0.965, "Network characteristics", transform = ax.transAxes, fontsize = 16, 
        fontweight = 'bold', horizontalalignment = 'center', verticalalignment = 'top')

props = dict(boxstyle = 'round', facecolor = 'white', edgecolor = 'black')
plt.text(0.86, 0.930, legend_info, transform = ax.transAxes, fontsize = 16, 
        verticalalignment = 'top', horizontalalignment = 'center', bbox = props)

plt.title('Training error vs Test error', fontsize = 20, fontweight = 'bold')


plt.tight_layout()

plt.tick_params(axis = 'x', labelsize = 16)
plt.tick_params(axis = 'y', labelsize = 16)
    

manager = plt.get_current_fig_manager()
manager.full_screen_toggle() 

plt.pause(2)

plt.savefig("grafici_per_slides/Monk2_accuracy.pdf", bbox_inches = 'tight', dpi = 1200)


plt.close()
plt.show()


# Accuracy graph

plt.figure()
plt.plot(accuracy_train, label = 'Training accuracy', linewidth = 2)
plt.plot(accuracy_test, label = 'Test accuracy', linewidth = 2)
plt.xlabel('Epochs', fontsize = 18, fontweight = 'bold')
plt.ylabel('Accuracy', fontsize = 18, fontweight = 'bold')
plt.grid()


plt.legend(labels = ['Training Accuracy', 'Test Accuracy'], fontsize = 25, loc = 'best')

plt.title('Training accuracy vs Test accuracy', fontsize = 20, fontweight = 'bold')

plt.tight_layout()

plt.tick_params(axis = 'x', labelsize = 16)
plt.tick_params(axis = 'y', labelsize = 16)

manager = plt.get_current_fig_manager()
manager.full_screen_toggle() 

plt.pause(2)

plt.savefig("grafici_per_slides/Monk2_accuracy.pdf", bbox_inches = 'tight', dpi = 1200)


plt.close()
plt.show()