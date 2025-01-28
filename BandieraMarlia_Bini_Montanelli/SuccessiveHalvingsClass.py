from itertools import product
import os
import time

from NeuralNetworkClass import NeuralNetwork
from Functions import activation_functions, d_activation_functions, loss_functions, d_loss_functions, activation_functions_grid, d_activation_functions_grid
from ModelSelectionClass import ModelSelection
from GridBuilding_adam import combinations_grid, x_trains, CUP_data_splitter, batch_size
# from GridBuilding_NAG_1 import combinations_grid, x_trains, CUP_data_splitter, batch_size
# from GridBuilding_NAG_2 import combinations_grid, x_trains, CUP_data_splitter, batch_size
from LossControlClass import LossControl

def print_nn_details(nn):
    '''
    Function that prints the Neural Network details.

    Args:
        nn (NeuralNetwork): instance of NeuralNetwork 
    '''
    details = []
    details.append("=== Neural Network Details ===")
    
    # Regolarizzazione
    details.append("\nRegularizer Configuration:")
    if hasattr(nn.regularizer, '__dict__'):
        for key, value in nn.regularizer.__dict__.items():
            details.append(f"  {key}: {value}")
    else:
        details.append("  No regularizer details available.")
    
    # Layer
    details.append("\nLayers Configuration:")
    for i, layer in enumerate(nn.layers):
        details.append(f"    Layer {i + 1}:")
        details.append(f"    dim_prev_layer: {layer.dim_prev_layer}")
        details.append(f"    dim_layer: {layer.dim_layer}")
        details.append(f"    activation_function: {layer.activation_function.__name__}")  # Nome della funzione
        details.append(f"    d_activation_function: {layer.d_activation_function.__name__}")  # Nome della funzione derivata
    
    # Ottimizzatori
    details.append("\nOptimizers Configuration:")
    allowed_optimizer_keys = ['opt_type', 'learning_rate', 'momentum', 'beta_1', 'beta_2', 'epsilon']
    for i, optimizer in enumerate(nn.optimizers):
        details.append(f"  Optimizer {i + 1}:")
        if hasattr(optimizer, '__dict__'):
            for key, value in optimizer.__dict__.items():
                if key in allowed_optimizer_keys:
                    details.append(f"    {key}: {value}")
        else:
            details.append(f"    Optimizer {i + 1} details not available.")
    
    return "\n".join(details)


def successive_halvings(list_combination, brackets, min_resources, max_resources):
    '''
    Function that performs successive halvings grid search on the hyperparameters

    Args:
        list_combination (list)
        brackets (int): number of splits being performed on the combinations of hyperparameter.
        min_resources (int): minimum number of epochs to perform training and validation before evaluating the set of hyperparameters.
        max_resources (int): maximum number of epochs.
    '''
    # training for a small number of epochs for all configurations
    resources = min_resources

    loss_control = LossControl(resources)
    
    # generating all configuration combinations
    for bracket in range(brackets):
        print(f"Bracket {bracket+1}/{brackets}")

        current_time = time.ctime()
        os.system('echo ' + current_time)

        # actual training
        results = []
        #a = 0

        for nn, batch in list_combination:
            #a+=1
            #if a%10 == 0:
            #    print(f'entra? {a}')

            train_val = ModelSelection(CUP_data_splitter, resources, batch, loss_functions['mse'], d_loss_functions['d_mse'], nn, loss_control)
            train_error_tot, val_error_tot = train_val.train_fold()
        
            # storing the result
            results.append({
                'nn': nn,
                'batch_size' : batch,
                'val_error': val_error_tot[-1],
                'train_error': train_error_tot[-1]  # validation error is used to evaluate the performance
            })
        
        # ordering the configurations according to their performance and keeping half of them
        results.sort(key = lambda x: x['val_error'])

        best_results = results[:len(results) // 2]

        # more resources for the best configurations        
        list_combination = list(zip([r['nn'] for r in best_results], [r['batch_size'] for r in best_results]))

        resources = min(resources * 2, max_resources)  
        
        # Controlla quanti elementi ci sono in best_configs
        num_elements = len(best_results)
        print(f'n element: {num_elements}')

    return best_results

#####################################################################################################################################

# Configurations building and successive halvings.

#####################################################################################################################################

'''
In this section of the code we take the combinations of GridBuilding in order to form the layer configuration, the regulizer configuration
and the optimizer configuration to give as parameters to the NeuralNetwork instance. Then it builds the actual neural networks. 
In order to perform the training algorithm, it calculates the combination of neural networks with the possible batch sizes.
'''

current_time = time.ctime()
os.system('echo ' + current_time)

# List of neural network combinations being investigated
nn_combo = []

# Layer configurations
used_combination = 0
for config in combinations_grid:
    # check if activation function and activation function derivative are the same
    if config['d_activation_function'].startswith('d_') and config['d_activation_function'][2:] == config['activation_function']:
        used_combination += 1
        
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

        # output layer
        layer_config.append({
                'dim_prev_layer': int(N_units[-1]),
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

        nn = NeuralNetwork(layer_config, reg_config, opt_config, True)
        nn_combo.append(nn)

combination = product(nn_combo, batch_size)
list_combination = list(combination)

used_combination = used_combination * len(batch_size)
print(f"Number of combination: {used_combination}")

'''
Definition of successive halvings parameters and application of the algorithm.
'''

# Successive Halvings parameters
brackets = 3  # number of brackets (times the number of configuration is reduced)
min_resources = 5  # min resources per configuration (= min epochs)
max_resources = 300  # max resources per configuration (= max epochs)

# Successive halvings application
best_configs = successive_halvings(list_combination, brackets, min_resources, max_resources)
    
'''
Writing a txt file with the combination selected
'''

with open("01_23_best_successivehalvings_configs_adam_grande.txt", "w") as file:
    for i in range(len(best_configs)):
        # Selection of the best combination
        final_best_result = best_configs[i]
        final_best_nn = final_best_result['nn']
        
        file.write(f"\n Best configuration after Successive Halvings n: {i+1} \n")
        file.write(f"Batch Size: {final_best_result['batch_size']}\n")
        file.write(f"Validation Error: {final_best_result['val_error']}\n")
        file.write("NN Details:\n")
        file.write(f"{print_nn_details(final_best_nn)}\n")
        file.write("\n" + "-"*50 + "\n")

current_time = time.ctime()
os.system('echo ' + current_time)