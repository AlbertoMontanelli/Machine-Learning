import numpy as np

from NeuralNetworkClass import NeuralNetwork
from Functions import activation_functions, d_activation_functions, loss_functions, d_loss_functions, activation_functions_grid, d_activation_functions_grid
from ModelSelectionClass import ModelSelection
from GridBuilding import combinations_grid, x_trains, CUP_data_splitter
from EarlyStoppingClass import EarlyStopping

def print_nn_details(nn):
    print("=== Neural Network Details ===")
    
    # Regolarizzazione
    print("\nRegularizer Configuration:")
    if hasattr(nn.regularizer, '__dict__'):
        for key, value in nn.regularizer.__dict__.items():
            print(f"  {key}: {value}")
    else:
        print("  No regularizer details available.")
    
    # Layer
    print("\nLayers Configuration:")
    for i, layer in enumerate(nn.layers):
        print(f"    Layer {i + 1}:")
        print(f"    dim_prev_layer: {layer.dim_prev_layer}")
        print(f"    dim_layer: {layer.dim_layer}")
        print(f"    activation_function: {layer.activation_function.__name__}")  # Nome della funzione
        print(f"    d_activation_function: {layer.d_activation_function.__name__}")  # Nome della funzione derivata
    
    # Ottimizzatori
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


# cose da fare:
# fare training con ogni configurazione di iperparametri per x epoche
# scartare tutte le opzioni tranne le migliori y
# portare a termine quelle y e valutare la metrica aggiungendo early stopping
# classifica
# selezione il set di iperparametri migliore

# List of neural network combinations being investigated
nn_combo = []

# Layer configurations
a = 0
b = 0
for config in combinations_grid:
    a+=1
    #print(f'entra? {a}')

    # controllo prima cosi se non sono uguali skippo direttamente
    if config['d_activation_function'].startswith('d_') and config['d_activation_function'][2:] == config['activation_function']:
        b += 1

        
        N_units = config['N_units']

        layer_config = []
        for i, units in enumerate(N_units):
            dim_prev_layer = []
            dim_layer = []

            layer_config.append({
                'dim_prev_layer': int(x_trains[0].shape[1]) if i == 0 else int(N_units[i - 1]),
                'dim_layer': int(units),
                'activation function': activation_functions_grid[config['activation_function']],
                'd_activation_function': d_activation_functions_grid[config['d_activation_function']]
            })
                
        # Aggiunta del layer di output

        layer_config.append({
                'dim_prev_layer': int(N_units[-1]), # dim dell'ultimo layer nascosto
                'dim_layer': 3,
                'activation function': activation_functions['linear'],
                'd_activation_function': d_activation_functions['d_linear']
            }
        )

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
        
print_nn_details(nn_combo[0])
print_nn_details(nn_combo[-1])
print(f'finite le iterazioni \n tutte: {a}, vere: {b}')

batch_size = [1, 16, len(x_trains[0])]
# batch_size = [1, 16, 32, 64, len(x_trains[0])]

# Hyperband parameters
brackets = 3  # number of brackets (times the number of configuration is reduced)
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

    early_stop = EarlyStopping(resources)
    
    # generating all configuration combinations
    for bracket in range(brackets):
        print(f"Bracket {bracket+1}/{brackets}")
        
        # actual training
        results = []
        a = 0
        for nn in nn_combo:
            a+=1
            if a%10 == 0:
                print(f'entra? {a}')
            

            for i in range(len(batch_size)):
                train_val = ModelSelection(CUP_data_splitter, resources, batch_size[i], loss_functions['mse'], d_loss_functions['d_mse'], nn, early_stop)
                train_error_tot, val_error_tot = train_val.train_fold(False, True)
            
                # storing the result
                results.append({
                    'nn': nn,
                    'batch_size' : batch_size[i],
                    'val_error': val_error_tot,
                    'train_error': train_error_tot  # validation error is used to evaluate the performance
                })
        
        # ordering the configurations according to their performance and keeping half of them
        results.sort(key = lambda x: x['val_error'])
        best_results = results[:len(results) // 2]

        # more resources for the best configurations
        nn_combo = [r['nn'] for r in best_results]  
        resources = min(resources * 2, max_resources)  

        # saving results for each bracket
        all_results.append(best_results)
        # Controlla quanti elementi ci sono in best_configs
        num_elements = len(all_results)
        print(f'n element: {num_elements}')

    return all_results


def Training_best_config(nn, batch_size, epochs):
    '''
    doc
    '''
    early_stop = EarlyStopping(epochs)
    train_val = ModelSelection(CUP_data_splitter, epochs, batch_size, loss_functions['mse'], d_loss_functions['d_mse'], nn, early_stop)
    train_error_tot, val_error_tot = train_val.train_fold(True, True)

    return train_error_tot[-1], val_error_tot[-1]


# hyperband application
best_configs = hyperband(nn_combo, brackets, min_resources, max_resources)
for i in range(0, 10, 1):
    print(f'quale? {i+1}')
    # selection of the best performing configuration
    final_best_result = best_configs[-1][i]  # La configurazione migliore (val_error minimo)
    
    final_best_nn = final_best_result['nn']

    # Print the best configuration's details along with batch_size and val_error
    print(f"\n Best configuration after Hyperband n: {i} \n")
    print(f"Batch Size: {final_best_result['batch_size']}")
    print(f"Validation Error: {final_best_result['val_error']}")
    print_nn_details(final_best_nn)

    train_error, val_error = Training_best_config(final_best_result['nn'], final_best_result['batch_size'], 500)
    print(f'ultimo error train: {train_error}')
    print(f'ultimo error val: {val_error}')