import numpy as np
import os
import time

from ModelSelectionClass import ModelSelection
from LossControlClass import LossControl
from CUPDataProcessing import CUP_data_splitter
from NeuralNetworkClass import NeuralNetwork
from Functions import activation_functions, d_activation_functions, loss_functions, d_loss_functions


def unpacking_config(config_text):
    """Parsa una singola configurazione dal testo."""
    config_dict = {}
    lines = config_text.split("\n")
    
    layers_config = []  # Lista per i layer
    current_layer = None
    
    for line in lines:
        line = line.strip()
        if line.startswith("Batch Size:"):
            config_dict["batch_size"] = int(line.split(":")[1].strip())
        elif line.startswith("Validation Error:"):
            config_dict["val_error"] = float(line.split(":")[1].strip())
        elif line.startswith("Lambda:"):
            config_dict.setdefault("regularization", {})["Lambda"] = float(line.split(":")[1].strip())
        elif line.startswith("alpha:"):
            config_dict["regularization"]["alpha"] = float(line.split(":")[1].strip())
        elif line.startswith("reg_type:"):
            config_dict["regularization"]["reg_type"] = line.split(":")[1].strip()
        elif line.startswith("dim_prev_layer:"):
            if current_layer:
                layers_config.append(current_layer)
            current_layer = [int(line.split(":")[1].strip())]  # Inizia un nuovo layer
        elif line.startswith("dim_layer:"):
            current_layer.append(int(line.split(":")[1].strip()))
        elif line.startswith("activation_function:"):
            func_name = line.split(":")[1].strip()
            current_layer.append(activation_functions[func_name])  # Usa il dizionario definito
        elif line.startswith("d_activation_function:"):
            func_name = line.split(":")[1].strip()
            current_layer.append(d_activation_functions[func_name])  # Usa il dizionario definito
        elif line.startswith("opt_type:"):
            config_dict.setdefault("optimizers", []).append({
                "opt_type": line.split(":")[1].strip()
            })
        elif line.startswith("learning_rate:"):
            config_dict["optimizers"][-1]["learning_rate"] = float(line.split(":")[1].strip())
        elif line.startswith("momentum:"):
            config_dict["optimizers"][-1]["momentum"] = float(line.split(":")[1].strip())
        elif line.startswith("beta_1:"):
            beta_1 = line.split(":")[1].strip()
            config_dict["optimizers"][-1]["beta_1"] = None if beta_1 == "None" else float(beta_1)
        elif line.startswith("beta_2:"):
            beta_2 = line.split(":")[1].strip()
            config_dict["optimizers"][-1]["beta_2"] = None if beta_2 == "None" else float(beta_2)
        elif line.startswith("epsilon:"):
            epsilon = line.split(":")[1].strip()
            config_dict["optimizers"][-1]["epsilon"] = None if epsilon == "None" else float(epsilon)

    # Aggiungi l'ultimo layer
    if current_layer:
        layers_config.append(current_layer)
    
    config_dict["layers_config"] = layers_config
    return config_dict



def training_best_config(nn, batch_size, epochs):
    '''
    doc
    '''
    loss_control = LossControl(epochs)
    train_val = ModelSelection(CUP_data_splitter, epochs, batch_size, loss_functions['mse'], d_loss_functions['d_mse'], nn, loss_control)
    train_error_tot, val_error_tot = train_val.train_fold(True)

    return train_error_tot[-1], val_error_tot[-1]


# Questo fa solo il print
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


# Leggi il file
with open("best_hyperband_configs_NAG_1.txt", "r") as file:
    lines = file.readlines()

# Processa le configurazioni
configs = []
current_config = []
separator = "--------------------------------------------------"

for line in lines:
    if line.strip() == separator:
        if current_config:
            config_text = "\n".join(current_config)
            configs.append(unpacking_config(config_text))
            current_config = []
    else:
        current_config.append(line.strip())

print(f"Caricate {len(configs)} configurazioni!")

# Training delle 30 migliori
for config in configs:

    print(config['layers_config'])
    print(config["regularization"])
    print(config["optimizers"])
    
    nn = NeuralNetwork(
        layers=config["layers_config"],
        regularization=config["regularization"],
        optimizers=config["optimizers"]
    )

    # Print the best configuration's details along with batch_size and val_error
    print(f"\n Best configuration after Hyperband\n")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Validation Error: {config['val_error']}")
    print_nn_details(nn)

    '''
    train_error, val_error = training_best_config(nn, config["batch_size"], 1000)
    print(f'ultimo error train: {train_error}')
    print(f'ultimo error val: {val_error}')
    '''
    
