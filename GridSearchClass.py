import numpy as np
from itertools import product
import os
import time

from NeuralNetworkClass import NeuralNetwork
from Functions import activation_functions, d_activation_functions, loss_functions, d_loss_functions, activation_functions_grid, d_activation_functions_grid
from ModelSelectionClass import ModelSelection
from GridBuilding_NAG_25_01_fine import combinations_grid, x_trains, CUP_data_splitter, batch_size
from LossControlClass import LossControl

#######################################################################################################################

# PRINT details nn

def print_nn_details(nn):
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

#######################################################################################################################

current_time = time.ctime()
os.system('echo ' + current_time)

# List of neural network combinations being investigated
nn_combo = []

# Layer configurations
all_combination = 0
used_combination = 0
for config in combinations_grid:
    all_combination += 1
    #print(f'entra? {all_combination}')

    # controllo prima cosi se non sono uguali skippo direttamente
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

        nn = NeuralNetwork(layer_config, reg_config, opt_config, True)
        
        nn_combo.append(nn)
        
print(f'finite le iterazioni tutte: {all_combination}, vere: {used_combination}')

combination = product(nn_combo, batch_size)
list_combination = list(combination)

epochs = 2000
loss_control = LossControl(epochs)


i = 0
results = []

for nn, batch in (list_combination):
    print(f'combinazione {i+1}')

    train_val = ModelSelection(CUP_data_splitter, epochs, batch, loss_functions['mee'], d_loss_functions['d_mee'], nn, loss_control)
    train_error_tot, val_error_tot, train_variance, val_variance, smoothness = train_val.train_fold(True, True, True)

    print_nn_details(nn)
    print(f'smoothness: {smoothness}')
    print(f'errore training {train_error_tot[-1]} +- {train_variance[-1]}')
    print(f'errore validation {val_error_tot[-1]} +- {val_variance[-1]}')
    results.append([train_error_tot, val_error_tot, smoothness, train_variance, val_variance, i+1])
    i = i+1
    print('\n')


print('\n')
print('\n')
print('Plot e salvataggio dei grafici')
print('\n')

import matplotlib.pyplot as plt


for i in range(len(results)):

    plt.figure()

    print(f'Configuration n {results[i][5]}, smoothness: {results[i][2]}')
    line_train, = plt.plot(results[i][0], label='Training Error')
    line_val, = plt.plot(results[i][1], label='Validation Error')

    plt.xlabel('Epochs', fontsize = 16, fontweight = 'bold')
    plt.ylabel('Error', fontsize = 16, fontweight = 'bold')
    plt.yscale('log')
    plt.grid()
    plt.legend(handles = [line_train, line_val], labels = ['Training Error', 'Validation Error'], fontsize = 18, loc = 'best')

    # Aggiungere padding tra i subplot
    plt.tight_layout()

    plt.tick_params(axis = 'x', labelsize = 16)  # Dimensione xticks
    plt.tick_params(axis = 'y', labelsize = 16)  # Dimensione yticks

    # Mettere il grafico a schermo intero
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle() 

    plt.pause(2)  # Pausa di 2 secondi

    # Salvare il grafico in PDF con alta risoluzione
    plt.savefig(f'grafici/config_NAG_25_01_fine_MEE_etagrande_{i+1}.pdf', bbox_inches = 'tight', dpi = 1200)

    plt.close()

    #plt.show()

j = 0
# Apri un file di testo in modalità scrittura
with open("config_NAG_25_01_fine_MEE_etagrande.txt", "w") as file:
    for nn, batch in (list_combination):
        # Seleziona la i-esima combinazione migliore
        
        # Scrivi i dettagli della configurazione migliore nel file
        file.write(f"\n Configuration n: {j+1} \n")
        file.write(f"Batch Size: {batch}\n")
        file.write(f"Validation Error: {results[j][1][-1]} +- {results[j][4][-1]} \n")
        file.write("NN Details:\n")
        file.write(f"{print_nn_details(nn)}\n")  # Supponendo che print_nn_details ritorni una stringa
        file.write("\n" + "-"*50 + "\n")
        j = j+1

current_time = time.ctime()
os.system('echo ' + current_time)