import pandas as pd

file_path = "monk+s+problems/ML-CUP24-TR.csv"

skip_lines = 7 # skipping metadata

first_column = ['Id']
last_columns = ['target_x', 'target_y', 'target_z']
middle_columns = [f'input_{i}' for i in range(1, 14)]

columns = first_column + middle_columns + last_columns
print(columns)
df = pd.read_csv(file_path, sep = ",", skiprows = skip_lines, names = columns, header = None)