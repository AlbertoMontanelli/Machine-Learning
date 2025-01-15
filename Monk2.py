import numpy as np
import matplotlib.pyplot as plt

from NeuralNetworkClass import NeuralNetwork
from Functions import activation_functions, d_activation_functions, loss_functions, d_loss_functions
from ModelAssessment import ModelAssessment
from MonkDataProcessing import monk_data

# Training of the neural network using monk2_data

# Layers configuration
np.random.seed(12)

layers_config = [
    (17, 10, activation_functions['sigmoid'], d_activation_functions['d_sigmoid']),
    (10, 1, activation_functions['sigmoid'], d_activation_functions['d_sigmoid'])
]

# Regularization configuration
reg_config = {
    'Lambda': 1e-4,
    'alpha' : 0.5,
    'reg_type': 'none'
}

# Optimization configuration
opt_config = {
    'opt_type': 'NAG',
    'learning_rate': 0.01,
    'momentum': 0.9,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'epsilon': 1e-8,
}


epochs = 500
batch_size = 20


########################################################################################################################

# MODEL ASSESSMENT FOR MONK1 DATASET WITH TRAINING, TEST ERROR CURVE AND TRAINING, TEST ACCURACY CURVE

########################################################################################################################

nn_assessment = NeuralNetwork(layers_config, reg_config, opt_config)

train_test = ModelAssessment(
    monk_data['training_set_2'], 
    monk_data['target_training_set_2'], 
    monk_data['test_set_2'], 
    monk_data['target_test_set_2'], 
    epochs, 
    batch_size, 
    loss_functions['bce'], 
    d_loss_functions['d_bce'], 
    nn_assessment)

train_error, test_error, accuracy_train, accuracy_test = train_test.retrain_test(accuracy_check = True)

# Creazione della figura con due subplot
fig, ax = plt.subplots(1, 2, figsize = (15, 10))

# Primo grafico (Training e Test Error)
line_train, = ax[0].plot(train_error, label = 'Training Error', linewidth = 2)
line_test, = ax[0].plot(test_error, label = 'Test Error', linewidth = 2)
ax[0].set_xlabel('Epochs', fontsize = 16, fontweight = 'bold')
ax[0].set_ylabel('Error', fontsize = 16, fontweight = 'bold')
ax[0].set_yscale('log')
ax[0].grid()

# Secondo grafico (Accuracy)
line_train_acc, = ax[1].plot(accuracy_train, label = 'Training accuracy', linewidth = 2)
line_test_acc, = ax[1].plot(accuracy_test, label = 'Test accuracy', linewidth = 2)
ax[1].set_xlabel('Epochs', fontsize = 16, fontweight = 'bold')
ax[1].set_ylabel('Accuracy', fontsize = 16, fontweight = 'bold')
ax[1].grid()

# Dati della rete da includere nella legenda
network_details = [
    ('Numbers of Hidden Layers', f'{layers_config[1][1]}'),
    ('Units per Layer', f'{layers_config[1][0]}'),
    ('Activation functions', 'sigmoid'),
    ('Loss function', 'bce'),
    ('Learning Rate', f'{opt_config['learning_rate']}'),
    ('Regularization', f'{reg_config['reg_type']}'),
    ('Optimizer',f'{opt_config['opt_type']}')
]

# Aggiungere informazioni della rete come stringa multilinea
legend_info = "\n".join([f"{param}: {value}" for param, value in network_details])

# Aggiungere le legende per le curve (Training, Test Error, Accuracy)
ax[0].legend(handles = [line_train, line_test], labels = ['Training Error', 'Test Error'], fontsize = 18)
ax[1].legend(handles = [line_train_acc, line_test_acc], labels = ['Training Accuracy', 'Test Accuracy'], 
             fontsize = 18, loc = 'best')


# Titolo del blocco
ax[1].text(0.71, 0.265, "Network characteristics", transform=ax[1].transAxes, fontsize=16, 
        fontweight='bold', horizontalalignment='center', verticalalignment='top')

# Aggiungere un rettangolo bianco dietro il testo per simulare la "legenda"
props = dict(boxstyle='round', facecolor='white', edgecolor='black')
ax[1].text(0.71, 0.23, legend_info, transform=ax[1].transAxes, fontsize=16, 
        verticalalignment='top', horizontalalignment='center', bbox=props)

# Titoli
ax[0].set_title('Error vs Epochs', fontsize = 18, fontweight = 'bold')
ax[1].set_title('Accuracy vs Epochs', fontsize = 18, fontweight = 'bold')

# Aggiungere padding tra i subplot
plt.tight_layout()

# Salvare il grafico in PDF con alta risoluzione
plt.savefig('grafici/monk2.pdf', bbox_inches='tight', pad_inches=0, dpi=1200)

# Mostrare il grafico
plt.show()