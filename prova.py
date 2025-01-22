import re

from Functions import *
from NeuralNetworkClass import NeuralNetwork

# Funzioni di attivazione disponibili
activation_functions_grid = {
    "tanh": tanh,
    "leaky_ReLU": leaky_relu,
    "linear": linear
}
d_activation_functions = {
    'd_leaky_ReLU': d_leaky_relu,
    'd_linear': d_linear,
    'd_tanh': d_tanh
}

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
            activation_function = activation_functions_grid.get(match[2], match[2])
            d_activation_function = d_activation_functions.get(match[3], match[3])
            layers_config.append((dim_prev_layer, dim_layer, activation_function, d_activation_function))

        # Parse Optimizers Configuration
        opt_match = re.search(r"opt_type:\s*(\S+)\n\s*learning_rate:\s*(\S+)\n\s*momentum:\s*(\S+)\n\s*beta_1:\s*(\S+)\n\s*beta_2:\s*(\S+)\n\s*epsilon:\s*(\S+)", raw_config)
        opt_config = {
            'opt_type': opt_match.group(1),
            'learning_rate': float(opt_match.group(2)),
            'momentum': float(opt_match.group(3)),
            'beta_1': float(opt_match.group(4)) if opt_match.group(4) != 'None' else None,
            'beta_2': float(opt_match.group(5)) if opt_match.group(5) != 'None' else None,
            'epsilon': float(opt_match.group(6)) if opt_match.group(6) != 'None' else None
        } if opt_match else {}

        # Salva la configurazione come tupla
        configurations.append((opt_config, reg_config, layers_config))

    return configurations

# Percorso al file txt
file_path = '/home/alberto-montanelli/Unipi/Git Repositories/Machine-Learning/best_hyperband_configs_NAG_1.txt'
configurations = parse_nn_configurations(file_path)

# Stampa la prima configurazione per verifica
print(configurations[0])

nn = NeuralNetwork(layers_config=configurations[0][2], reg_config=configurations[0][1], opt_config=configurations[0][0])



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

print_nn_details(nn)
