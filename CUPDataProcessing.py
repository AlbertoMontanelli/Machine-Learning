import pandas as pd
import numpy as np

from DataProcessingClass import DataProcessing

file_path = "ML-CUP24-TR.csv"

skip_lines = 7 # skipping metadata

first_column = ['Id']
last_columns = ['target_x', 'target_y', 'target_z']
middle_columns = [f'input_{i}' for i in range(1, 13)]

columns = first_column + middle_columns + last_columns
df = pd.read_csv(file_path, sep = ",", skiprows = skip_lines, names = columns, header = None)
#print(f"df: \n {df.iloc[0]}")

data = df.drop(columns = ['Id', 'target_x', 'target_y', 'target_z'])
print(f"data: \n {data}")
target = df.drop(columns = [f'input_{i}' for i in range(1, 13)])
target = target.drop(columns = ['Id'])
print(f"target: \n {target}")


data = data.to_numpy()
target = target.to_numpy()

print(f"data: \n {data}")
print(f"target: \n {target}")

CUP_data_splitter = DataProcessing(data, target, 0.2, K=6)

