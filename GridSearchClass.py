import numpy as np

from NeuralNetworkClass import NeuralNetwork
from Functions import activation_functions, d_activation_functions, loss_functions, d_loss_functions, activation_functions_grid, d_activation_functions_grid
from ModelSelectionClass import ModelSelection
from GridBuilding import combinations_grid, x_trains, target_trains, x_vals, target_vals, CUP_data_splitter

# cose da fare:
# fare training con ogni configurazione di iperparametri per x epoche
# scartare tutte le opzioni tranne le migliori y
# portare a termine quelle y e valutare la metrica aggiungendo early stopping
# classifica
# selezione il set di iperparametri migliore

# List of neural network combinations being investigated
nn_combo = []

# Layer configurations
for config in combinations_grid:
    N_layer = config['N_layer']
    N_units = config['N_units']

    layer_config = []
    for i, units in enumerate(N_units):
        dim_prev_layer = []
        dim_layer = []

        if config['d_activation_function'].startswith('d_') and config['d_activation_function'][2:] == config['activation_function']: 
            layer_config.append({
                'dim_prev_layer': int(x_trains[0].shape[1]) if i == 0 else int(N_units[i - 1]),
                'dim_layer': int(units),
                'activation function': activation_functions_grid[config['activation_function']],
                'd_activation_function': d_activation_functions_grid[config['d_activation_function']]
            })


    # Regularization configurations
    reg_config = ({
        'Lambda': config['lambda'],
        'alpha': config['alpha'],
        'reg_type': 'elastic'
    })

    # Optimization configurations
    opt_config = ({
        'opt_type': config['opt_type'],
        'learning_rate': config['learning_rate'],
        'momentum': 0.9 if config['opt_type'] == 'NAG' else None,
        'beta_1': 0.9 if config['opt_type'] == 'adam' else None,
        'beta_2': 0.999 if config['opt_type'] == 'adam' else None,
        'epsilon': 1e-8 if config['opt_type'] == 'adam' else None
    })

    nn = NeuralNetwork(layer_config, reg_config, opt_config)
    nn_combo.append(nn)

batch_size = [1, 16, 32, 64, len(x_trains[0])]

# Hyperband parameters
brackets = 3  # number of brackets (times the number of configuration is reduceds)
min_resources = 1  # min resources per configuration (= min epochs)
max_resources = 300  # max resources per configuration (= max epochs)


def hyperband(nn_combo, brackets, min_resources, max_resources):
    '''
    Function that performs hyperband grid search on the hyperparameters

    Args:
        nn_combo (dict): dictionary of the neural network combinations.
        brackets (int): number of splits being performed on the combinations of hyperparameter.
        min_resources (int): minimum number of epochs to perform training and validation before evaluating the set of hyperparameters.
        max_resources (int): maximum number of epochs.
    '''
    # training for a small number of epochs for all configurations
    resources = min_resources
    all_results = []
    
    # generating all configuration combinations
    for bracket in range(brackets):
        print(f"Bracket {bracket+1}/{brackets}")
        
        # actual training
        results = []
        for nn in nn_combo:
            train_errors = []
            val_errors = []
            for i in range(len(batch_size)):
                train_val = ModelSelection(CUP_data_splitter, resources, batch_size[i], loss_functions['mse'], d_loss_functions['d_mse'], nn, early_stop=True)
                train_error_tot, val_error_tot = train_val.train_fold()
                train_errors.append(train_error_tot)
                val_errors.append(val_error_tot)
            
            # storing the result
            results.append({
                'nn': nn,
                'batch_size' : batch_size[i],
                'val_error': np.mean(val_errors)  # validation error is used to evaluate the performance
            })
        
        # ordering the configurations according to their performance and keeping half of them
        results.sort(key=lambda x: x['val_error'])
        best_results = results[:len(results) // 2]

        # more resources for the best configurations
        nn_combo = [r['nn'] for r in best_results]  
        resources = min(resources * 2, max_resources)  

        # saving results for each bracket
        all_results.append(best_results)

    return all_results

# hyperband application
best_configs = hyperband(nn_combo, brackets, min_resources, max_resources)

# selection of the best performing configuration
final_best_nn = best_configs[-1][0]['nn']
print(f"Best configuration after Hyperband: {final_best_nn}")


