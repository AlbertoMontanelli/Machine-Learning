import numpy as np
import pandas as pd

from DataProcessingClass import DataProcessing

file_path = "monk+s+problems/monks-1.train"
columns = ["target", "feat1", "feat2", "feat3", "feat4", "feat5", "feat6", "Id"]  
df = pd.read_csv(file_path, sep = " ", header = None, names = columns, skipinitialspace = True)
print(df)
training_set1 = df.drop(columns = ["target", "Id"])
print(training_set1)
training_set1 = pd.get_dummies(training_set1, columns = training_set1.columns[1:], drop_first = False, dtype = int)
target_set1 = df.drop(columns = ["feat1", "feat2", "feat3", "feat4", "feat5", "feat6", "Id"])

Datapepper = DataProcessing(training_set1, target_set1, test_perc = 0., K = 5)


