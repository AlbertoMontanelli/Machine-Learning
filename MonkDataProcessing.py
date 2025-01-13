import pandas as pd

# Importing data for all monk datasets
columns = ["target", "feat1", "feat2", "feat3", "feat4", "feat5", "feat6", "Id"]

for i in range(1, 4, 1):
    file_path_train = f"monk+s+problems/monks-{i}.train"
    file_path_test = f"monk+s+problems/monks-{i}.test"
    df_train = pd.read_csv(file_path_train, sep = " ", header = None, names = columns, skipinitialspace = True)
    df_test = pd.read_csv(file_path_test, sep = " ", header = None, names = columns, skipinitialspace = True)

    # Separation between targets and data. Last column is dropped because useless.
    training_set = df_train.drop(columns = ["target", "Id"])
    target_training_set = df_train["target"]
    
    test_set = df_test.drop(columns = ["target", "Id"])
    target_test_set = df_test["target"]

    # One-hot encoding of training, validation and test set for data.
    training_set = pd.get_dummies(training_set, columns = training_set.columns[:], drop_first = False, dtype = int)
    test_set = pd.get_dummies(test_set, columns = test_set.columns[:], drop_first = False, dtype = int)

    # Conversion from pandas.dataframe to numpy arrays.
    training_set = training_set.to_numpy()
    target_training_set = target_training_set.to_numpy()
    target_training_set = target_training_set.reshape(-1, 1)

    test_set = test_set.to_numpy()
    target_test_set = target_test_set.to_numpy()
    target_test_set= target_test_set.reshape(-1, 1)

    # Saving all the data in different global variables for each monk dataset.
    globals()[f"training_set_{i}"] = training_set
    globals()[f"target_training_set_{i}"] = target_training_set
    globals()[f"test_set_{i}"] = test_set
    globals()[f"target_test_set_{i}"] = target_test_set
