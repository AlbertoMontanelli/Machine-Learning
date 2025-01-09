import numpy as np

from DataProcessingClass import DataProcessing
from NeuralNetworkClass import NeuralNetwork
from Functions import activation_functions, d_activation_functions, loss_functions, d_loss_functions
from TrainingValidationClass import TrainingValidation

'''
IPERPARAMETRI

numero epoche forse no

NR DI LAYER
NR DI UNITÀ PER LAYER
ACTIVATION_FUNCTION
numero batch
LEARNING RATE_W
LEARNING RATE_B
LAMBDA_T
LAMBDA_L
ALPHA
MOMENTUM
BETA_1
BETA_2
'''

param_grid = {
    'N_layer' : [1, 2, 3],
    'N_units' : [16, 32, 64],
    'activation_function' : list(activation_functions.keys()),
    'batch_size' : [1, 16, 32, 64], #includere anche il totale dei dati
    'learning_rate_w' : np.logspace(-4, -2, num=5),  # 1e-4, 3.16e-4, ..., 1e-2
    'learning_rate_b' : np.logspace(-4, -2, num=5),  # si puà usare anche un linspace
    'reg_type': ['tikhonov', 'lasso', 'elastic'],  # Specifica il tipo di regolarizzazione
    'lambda_t': [1e-4, 1e-3] if 'tikhonov' in ['tikhonov', 'elastic'] else [0],  # Lambda_t solo per Tikhonov/Elastic
    'lambda_l': [1e-4, 1e-3] if 'lasso' in ['lasso', 'elastic'] else [0],       # Lambda_l solo per Lasso/Elastic
    'alpha': [0.1, 0.5, 0.9] if 'elastic' in ['elastic'] else [0],              # Alpha solo per Elastic Net
    'opt_type': ['adam', 'NAG'], 
    'momentum': [0.9, 0.99] if 'NAG' else [0],  # Momentum solo per NAG
    'beta_1': [0.9, 0.95] if 'adam' else [0],  # Beta_1 solo per Adam
    'beta_2': [0.99, 0.999] if 'adam' else [0],  # Beta_2 solo per Adam
}

#################################################################################################
#  DA RIVEDERE QUESTA ROBA PER NON SELEZIONARE GLI IPERPARAMETRI QUANDO QUESTI NON VENGONO USATI
import itertools

param_grid = {
    'reg_type': ['tikhonov', 'lasso', 'elastic'],
    'lambda_t': [1e-4, 1e-3],  # Solo per 'tikhonov' o 'elastic'
    'lambda_l': [1e-4, 1e-3],  # Solo per 'lasso' o 'elastic'
    'alpha': [0.1, 0.5],       # Solo per 'elastic'
    'learning_rate_w': [1e-4, 1e-3],
    'learning_rate_b': [1e-4, 1e-3],
}

# Genera tutte le combinazioni
all_combinations = list(itertools.product(*param_grid.values()))

# Filtra combinazioni non valide
valid_combinations = [
    dict(zip(param_grid.keys(), values))
    for values in all_combinations
    if (values[0] == 'tikhonov' and values[1] is not None and values[2] is None and values[3] == 0.1)
    or (values[0] == 'lasso' and values[1] is None and values[2] is not None and values[3] == 0.1)
    or (values[0] == 'elastic' and values[1] is not None and values[2] is not None and values[3] != 0)
]

print(valid_combinations)
############################################################################################################

class GridSearch:

    def __init__(self):
        '''
        
        '''
        pass


