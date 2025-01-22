import re
from Functions import activation_functions, d_activation_functions
from NeuralNetworkClass import NeuralNetwork

import re
from Functions import activation_functions, d_activation_functions

def normalize_key(key, dictionary):
    """
    Normalize a key to match the dictionary.
    If not found, raise a KeyError.
    """
    normalized_key = key.lower().replace("_relu", "_ReLU")
    if normalized_key not in dictionary:
        raise KeyError(f"Key '{key}' not found in the dictionary.")
    return dictionary[normalized_key]

def parse_configurations(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    
    # Split configurations by the delimiter line
    configurations = content.split("-" * 30)
    
    parsed_configs = []
    
    for config in configurations:
        if not config.strip():
            continue  # Skip empty parts

        # Regularizer Configuration
        reg_config_pattern = r"Regularizer Configuration:\s+Lambda: ([\d.e-]+)\s+alpha: ([\d.e-]+)\s+reg_type: (\w+)"
        reg_config_match = re.search(reg_config_pattern, config)
        reg_config = {
            "Lambda": float(reg_config_match.group(1)),
            "alpha": float(reg_config_match.group(2)),
            "reg_type": reg_config_match.group(3)
        }

        # Layers Configuration
        layers_config = []
        layers_pattern = r"Layer \d+:\s+dim_prev_layer: (\d+)\s+dim_layer: (\d+)\s+activation_function: (\w+)\s+d_activation_function: (\w+)"
        for match in re.finditer(layers_pattern, config):
            dim_prev_layer, dim_layer, activation, d_activation = match.groups()
            layers_config.append((
                int(dim_prev_layer),
                int(dim_layer),
                normalize_key(activation, activation_functions),
                normalize_key(d_activation, d_activation_functions)
            ))

        # Optimizer Configuration
        opt_config_pattern = (
            r"Optimizers Configuration:\s+Optimizer \d+:\s+opt_type: (\w+)\s+"
            r"learning_rate: ([\d.e-]+)\s+momentum: ([\d.e-]+)\s+"
            r"beta_1: ([\d.e-]+|None)\s+beta_2: ([\d.e-]+|None)\s+epsilon: ([\d.e-]+|None)"
        )
        opt_config_match = re.search(opt_config_pattern, config)
        opt_config = {
            "opt_type": opt_config_match.group(1),
            "learning_rate": float(opt_config_match.group(2)),
            "momentum": float(opt_config_match.group(3)),
            "beta_1": float(opt_config_match.group(4)) if opt_config_match.group(4) != "None" else None,
            "beta_2": float(opt_config_match.group(5)) if opt_config_match.group(5) != "None" else None,
            "epsilon": float(opt_config_match.group(6)) if opt_config_match.group(6) != "None" else None
        }
        
        # Append the parsed configuration
        parsed_configs.append({
            "layers_config": layers_config,
            "reg_config": reg_config,
            "opt_config": opt_config
        })
    
    return parsed_configs

# Path to the configuration file
file_path = "best_hyperband_configs_NAG_2.txt"

# Parse all configurations
parsed_configs = parse_configurations(file_path)

# Initialize neural networks for each configuration
neural_networks = []
for config in parsed_configs:
    nn = NeuralNetwork(config["layers_config"], config["reg_config"], config["opt_config"])
    neural_networks.append(nn)

# Output summary
for idx, config in enumerate(parsed_configs):
    print(f"Configuration {idx + 1}:")
    print("Layers Config:", config["layers_config"])
    print("Regularizer Config:", config["reg_config"])
    print("Optimizer Config:", config["opt_config"])
    print()




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

print_nn_details(neural_networks)
