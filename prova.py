

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
            activation_function = activation_functions.get(match[2], match[2])
            d_activation_function = d_activation_functions.get(match[3], match[3])
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
file_path = 'best_hyperband_configs_adam.txt'
configurations = parse_nn_configurations(file_path)

# Stampa la prima configurazione per verifica
print(configurations[0])

neural_networks = []
for i in range(len(configurations)):
    nn = NeuralNetwork(layers_config=configurations[i][2], reg_config=configurations[i][1], opt_config=configurations[i][0])
    neural_networks.append(nn)


epochs = 500
loss_control = LossControl(epochs)


total_config = np.zeros(len(neural_networks))


for nn, i in neural_networks, len(neural_networks):

    train_val = ModelSelection(CUP_data_splitter, epochs, configurations[i][4], loss_functions['mse'], d_loss_functions['d_mse'], nn, loss_control)
    train_error_tot, val_error_tot, smoothness = train_val.train_fold(True)
   
    total_config[i] = tuple(nn, smoothness, train_error_tot, val_error_tot)
    print(f'combinazione {i+1} \n ')
    print_nn_details(nn)
    print(f'smoothness: {smoothness}')
    print(f'errore training {train_error_tot[-1]}')
    print(f'errore validation {val_error_tot[-1]}')

