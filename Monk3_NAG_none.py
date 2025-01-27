import numpy as np

from NeuralNetworkClass import NeuralNetwork
from Functions import activation_functions, d_activation_functions, loss_functions, d_loss_functions
from ModelAssessmentClass import ModelAssessment
from LossControlClass import LossControl

from MonkDataProcessing import monk_data

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

# Layers configuration
layers_config = [
    (17, 4, activation_functions['sigmoid'], d_activation_functions['d_sigmoid']),
    (4, 1, activation_functions['sigmoid'], d_activation_functions['d_sigmoid'])
]

# Regularization configuration
reg_config_NAG = {
    'Lambda': 1e-5,
    'alpha' : 0.5,
    'reg_type': 'elastic'
}

# Optimization configuration
opt_config_NAG = {
    'opt_type': 'NAG',
    'learning_rate': 5e-3,
    'momentum': 0.9,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'epsilon': 1e-8,
}

# Regularization configuration
reg_config_none = {
    'Lambda': 5e-4,
    'alpha' : 0.5,
    'reg_type': 'none'
}

# Optimization configuration
opt_config_none = {
    'opt_type': 'none',
    'learning_rate': 5e-3,
    'momentum': 0.9,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'epsilon': 1e-8,
}

epochs = 500
batch_size = 1


########################################################################################################################

# MODEL ASSESSMENT FOR MONK3 DATASET WITH TRAINING, TEST ERROR CURVE AND TRAINING, TEST ACCURACY CURVE

########################################################################################################################

nn_NAG = NeuralNetwork(layers_config, reg_config_NAG, opt_config_NAG)

np.random.seed(12)
nn_none = NeuralNetwork(layers_config, reg_config_none, opt_config_none)

loss_control = LossControl(epochs)

train_test_NAG = ModelAssessment(
    monk_data['training_set_3'], 
    monk_data['target_training_set_3'], 
    monk_data['test_set_3'], 
    monk_data['target_test_set_3'], 
    epochs, 
    batch_size, 
    loss_functions['mse'], 
    d_loss_functions['d_mse'], 
    nn_NAG,
    loss_control,
    classification_problem = True
    )

train_test_none = ModelAssessment(
    monk_data['training_set_3'], 
    monk_data['target_training_set_3'], 
    monk_data['test_set_3'], 
    monk_data['target_test_set_3'], 
    epochs, 
    batch_size, 
    loss_functions['mse'], 
    d_loss_functions['d_mse'], 
    nn_none,
    loss_control,
    classification_problem = True
    )

train_error_NAG, test_error_NAG, accuracy_train_NAG, accuracy_test_NAG = train_test_NAG.retrain_test()
train_error_none, test_error_none, accuracy_train_none, accuracy_test_none = train_test_none.retrain_test()


########################################################################################################################

# PLOT

########################################################################################################################

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter

# Figure
plt.figure()

# Loss functions graph
plt.plot(train_error_NAG, label = 'Training Error Adam', linewidth = 2, color = 'cornflowerblue', linestyle = '-')
plt.plot(test_error_NAG, label = 'Test Error Adam', linewidth = 2, color = 'sandybrown', linestyle = '-')
plt.plot(train_error_none, label = 'Training Error no opt', linewidth = 2, color = 'cornflowerblue', linestyle = '--')
plt.plot(test_error_none, label = 'Test Error no opt', linewidth = 2, color = 'sandybrown', linestyle = '--')
plt.xlabel('Epochs', fontsize = 18, fontweight = 'bold')
plt.ylabel('MSE Error', fontsize = 18, fontweight = 'bold')
plt.yscale('log')

ax = plt.gca()  # Ottieni gli assi correnti
ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
ax.yaxis.set_major_formatter(ScalarFormatter())

plt.grid()



# Creation of a legend containing the details of the neural network
network_details = [
    ('Number of Hidden Layers', f'{layers_config[1][1]}'),
    ('Units per Layer', f'{layers_config[1][0]}'),
    ('Activation function', 'sigmoid'),
    ('Loss function', 'MSE'),
    ('Batch size', f"{batch_size}"),
    ('Learning Rate', f"{opt_config_none['learning_rate']}"),
    ('For optimizer',f"{opt_config_NAG['opt_type']}"),
    ('Momentum', f"{opt_config_NAG['momentum']}"),
    ('Regularization', f"{reg_config_NAG['reg_type']}"),
    ('lambda', f"{reg_config_NAG['Lambda']}"),
    ('alpha', f"{reg_config_NAG['alpha']}"),
    ('For optimizer',f"{opt_config_none['opt_type']}"),
    ('Regularization', f"{reg_config_none['reg_type']}"),
    #('lambda', f"{reg_config_none['Lambda']}"),
    #('alpha', f"{reg_config_none['alpha']}"),
]

legend_info = "\n".join([f"{param}: {value}" for param, value in network_details])
plt.legend(labels = ['Training Error NAG', 'Test Error NAG', 'Training Error no opt', 'Test Error no opt'], fontsize = 18, loc = 'best')



plt.text(0.86, 0.965, "Network characteristics", transform = ax.transAxes, fontsize = 16, 
        fontweight = 'bold', horizontalalignment = 'center', verticalalignment = 'top')

props = dict(boxstyle = 'round', facecolor = 'white', edgecolor = 'black')
plt.text(0.86, 0.930, legend_info, transform = ax.transAxes, fontsize = 16, 
        verticalalignment = 'top', horizontalalignment = 'center', bbox = props)

plt.title('Training error and Test error for adam, no opt.', fontsize = 20, fontweight = 'bold')


plt.tight_layout()

plt.tick_params(axis = 'x', labelsize = 16)
plt.tick_params(axis = 'y', labelsize = 16)
    

manager = plt.get_current_fig_manager()
manager.full_screen_toggle() 

plt.pause(2)

plt.savefig("grafici_per_slides/Monk3_error_NAG_none.pdf", bbox_inches = 'tight', dpi = 1200)


plt.close()
plt.show()


# Accuracy graph

plt.figure()
plt.plot(accuracy_train_NAG, label = 'Training accuracy NAG', linewidth = 2, color = 'cornflowerblue', linestyle = '-')
plt.plot(accuracy_test_NAG, label = 'Test accuracy NAG', linewidth = 2, color = 'sandybrown', linestyle = '-')
plt.plot(accuracy_train_none, label = 'Training accuracy no opt', linewidth = 2, color = 'cornflowerblue', linestyle = '--')
plt.plot(accuracy_test_none, label = 'Test accuracy no opt', linewidth = 2, color = 'sandybrown', linestyle = '--')
plt.xlabel('Epochs', fontsize = 18, fontweight = 'bold')
plt.ylabel('Accuracy', fontsize = 18, fontweight = 'bold')
plt.grid()


plt.legend(labels = ['Training Accuracy NAG', 'Test Accuracy NAG', 'Training Accuracy no opt', 'Test Accuracy no opt'], fontsize = 25, loc = 'best')

plt.title('Training accuracy and Test accuracy for NAG, no opt.', fontsize = 20, fontweight = 'bold')

plt.tight_layout()

plt.tick_params(axis = 'x', labelsize = 16)
plt.tick_params(axis = 'y', labelsize = 16)

manager = plt.get_current_fig_manager()
manager.full_screen_toggle() 

plt.pause(2)

plt.savefig("grafici_per_slides/Monk3_accuracy_NAG_none.pdf", bbox_inches = 'tight', dpi = 1200)


plt.close()
plt.show()