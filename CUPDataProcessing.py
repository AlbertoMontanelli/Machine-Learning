import pandas as pd
import numpy as np


from DataProcessingClass import DataProcessing

# Importing labelled data
file_path_labelled = "ML-CUP24-TR.csv"

skip_lines = 7 # skipping metadata

id_column = ['Id']
target_columns = ['target_x', 'target_y', 'target_z']
input_columns = [f'input_{i}' for i in range(1, 13)]
columns = id_column + input_columns + target_columns

df = pd.read_csv(file_path_labelled, sep = ",", skiprows = skip_lines, names = columns, header = None)
data = df.drop(columns = ['Id', 'target_x', 'target_y', 'target_z'])
target = df.drop(columns = ['Id'] + [f'input_{i}' for i in range(1, 13)])

# Conversion to numpy arrays
data = data.to_numpy()
target = target.to_numpy()

# Splitting the CUP labelled data into training set, validation set and test set.
# A K-Fold Cross Validation with K = 6 will be performed during Model Selection.
CUP_data_splitter = DataProcessing(data, target, test_perc = 0.2, K = 5)

# Importing blind data
file_path_blind = "ML-CUP24-TS.csv"

columns_blind = id_column + input_columns

df_blind = pd.read_csv(file_path_blind, sep = ",", skiprows = skip_lines, names = columns_blind, header = None)
data_blind = df_blind.drop(columns = ['Id'])

# Conversion to numpy arrays
data_blind = data_blind.to_numpy()


'''
# printing
print(f'data \n {data}')
print(f'target \n {target}')
print(f'blind data \n {data_blind}')
'''