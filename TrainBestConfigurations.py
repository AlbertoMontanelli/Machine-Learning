import re
import numpy as np

from Functions import activation_functions, d_activation_functions, loss_functions, d_loss_functions
from NeuralNetworkClass import NeuralNetwork
from ModelSelectionClass import ModelSelection
from LossControlClass import LossControl
from CUPDataProcessing import CUP_data_splitter

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

def parse_nn_configurations(file_path):
    configurations = []

    # Normalizzazione dei dizionari per rendere le chiavi case-insensitive
    activation_functions_lower = {key.lower(): value for key, value in activation_functions.items()}
    d_activation_functions_lower = {key.lower(): value for key, value in d_activation_functions.items()}

    # Leggi il file
    with open(file_path, 'r') as file:
        data = file.read()

    # Dividi le configurazioni separate da "--------------------------------------------------"
    raw_configs = data.split('--------------------------------------------------')

    for raw_config in raw_configs:
        if not raw_config.strip():
            continue

        # Parse Batch Size
        batch_size_match = re.search(r"Batch Size:\s*(\d+)", raw_config)
        batch_size = int(batch_size_match.group(1)) if batch_size_match else None

        # Parse Regularizer Configuration
        reg_match = re.search(r"Lambda:\s*(\S+)\n\s*alpha:\s*(\S+)\n\s*reg_type:\s*(\S+)", raw_config)
        reg_config = {
            'Lambda': float(reg_match.group(1)),
            'alpha': float(reg_match.group(2)),
            'reg_type': reg_match.group(3)
        } if reg_match else {}

        # Parse Layers Configuration
        layers_config = []
        layer_matches = re.findall(r"Layer \d+:\n\s*dim_prev_layer:\s*(\d+)\n\s*dim_layer:\s*(\d+)\n\s*activation_function:\s*(\S+)\n\s*d_activation_function:\s*(\S+)", raw_config)
        for match in layer_matches:
            dim_prev_layer = int(match[0])
            dim_layer = int(match[1])
            
            try:
                activation_function = activation_functions_lower[match[2].lower()]
                d_activation_function = d_activation_functions_lower[match[3].lower()]
            except KeyError as e:
                raise ValueError(f"Funzione di attivazione non trovata: {e}")

            layers_config.append((dim_prev_layer, dim_layer, activation_function, d_activation_function))

        # Parse Optimizers Configuration
        opt_match = re.search(r"opt_type:\s*(\S+)\n\s*learning_rate:\s*(\S+)\n\s*momentum:\s*(\S+)\n\s*beta_1:\s*(\S+)\n\s*beta_2:\s*(\S+)\n\s*epsilon:\s*(\S+)", raw_config)
        opt_config = {
            'opt_type': opt_match.group(1),
            'learning_rate': float(opt_match.group(2)),
            'momentum': float(opt_match.group(3)) if opt_match.group(3) != 'None' else None,
            'beta_1': float(opt_match.group(4)) if opt_match.group(4) != 'None' else None,
            'beta_2': float(opt_match.group(5)) if opt_match.group(5) != 'None' else None,
            'epsilon': float(opt_match.group(6)) if opt_match.group(6) != 'None' else None
        } if opt_match else {}

        # Salva la configurazione come tupla
        configurations.append((opt_config, reg_config, layers_config, batch_size))

    return configurations


# Percorso al file txt
file_path = '01_23_best_hyperband_configs_adam_grande.txt'
configurations = parse_nn_configurations(file_path)

# Stampa la prima configurazione per verifica
print(configurations[0])

neural_networks = []
for i in range(len(configurations)):
    nn = NeuralNetwork(layers_config=configurations[i][2], reg_config=configurations[i][1], opt_config=configurations[i][0])
    neural_networks.append(nn)


epochs = 500
loss_control = LossControl(epochs)

total_config = []

i = 3
while (len(total_config) <= 10):
    nn = neural_networks[i]
    train_val = ModelSelection(CUP_data_splitter, epochs, configurations[i][3], loss_functions['mse'], d_loss_functions['d_mse'], nn, loss_control)
    train_error_tot, val_error_tot, smoothness = train_val.train_fold(True, True, True)
    print(f'combinazione {i+1}')
    print(f'smoothness: {smoothness}')
    print(f'errore training {train_error_tot[-1]}')
    print(f'errore validation {val_error_tot[-1]}')
    if smoothness == True:
        print(f'appesa combinazione {i+1}')
        total_config.append([nn, smoothness, train_error_tot, val_error_tot])
    i = i+1
    print('\n')


print('\n')
print('\n')
print('Plot e salvataggio dei grafici')
print('\n')

import matplotlib.pyplot as plt


for i in range(len(total_config)):

    plt.figure()

    print(f'best configuration n {i}, smoothness: {total_config[i][1]}')
    line_train, = plt.plot(total_config[i][2], label='Training Error')
    line_val, = plt.plot(total_config[i][3], label='Validation Error')

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
    plt.savefig(f'grafici/01_23_best_adam_grande_{i}.pdf', bbox_inches = 'tight', dpi = 1200)

    plt.close()

    #plt.show()