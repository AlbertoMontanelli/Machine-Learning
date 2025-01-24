import pandas as pd

from DataProcessingClass import DataProcessing

'''
    Code that:
        extracts data from .csv files;
        splits the CUP labelled data into training set, validation set and test set;
        processes the CUP unlabelled data (blind data);
        converts pandas dataframes in numpy arrays.
    A K-Fold Cross Validation with K = 5 will be performed during Model Selection.
    At the end of this code, we have:
        from ML-CUP24-TR.csv:
            CUP_data_splitter.x_trains;
            CUP_data_splitter.target_trains;
            CUP_data_splitter.x_vals;
            CUP_data_splitter.target_vals;
            CUP_data_splitter.x_test;
            CUP_data_splitter.target_test.
        from ML-CUP24-TS.csv:
            data_blind.
'''

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

# CUP_data_splitter is an instance of DataProcessingClass
CUP_data_splitter = DataProcessing(data, target, test_perc = 0.2, K = 5)

# Importing blind data
file_path_blind = "ML-CUP24-TS.csv"
columns_blind = id_column + input_columns

df_blind = pd.read_csv(file_path_blind, sep = ",", skiprows = skip_lines, names = columns_blind, header = None)
data_blind = df_blind.drop(columns = ['Id'])

# Conversion to numpy arrays
data_blind = data_blind.to_numpy()