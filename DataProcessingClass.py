import numpy as np

class DataProcessing:

    def __init__(self, x_tot, target_tot):
        '''
        Class focused on processing the data into Training Set, Validation Set and Test Set

        Args:
            x_tot (array): total data given as input.
            target_tot (array): total data labels given as input.
        '''

        self.x_tot = x_tot
        self.target_tot = target_tot

    def test_split(self, test_perc):
        '''
        Function that splits the input data into two sets: training & validation set, test set.
        
        Args:
            test_perc (float): percentile of test set with respect to the total data.
        
        Returns:
            x_train_val (array): training and validation set extracted from input data.
            target_train_val (array): training and validation set labels.
            x_test_val (array): test set extracted from input data.
            target_test_val (array): test set for input data labels.
        '''

        if (test_perc > 1 or test_perc < 0):
            raise ValueError(f"Invalid {test_perc}. Choose from 0 to 1")
        
        num_samples = self.x_tot.shape[0] # the total number of the examples in input is the number of rows in the x_tot matrix
        test_size = int(num_samples * test_perc) # the number of the examples in the test set

        x_test = self.x_tot[:test_size]
        target_test = self.target_tot[:test_size]
        x_train_val = self.x_tot[test_size:]
        target_train_val = self.target_tot[test_size:]

        return x_train_val, target_train_val, x_test, target_test
    
    
    def train_val_split(self, test_perc, K = 1, train_perc = 0.75):
        '''
        Function that splits the training & validation set into two sets: training set and validation set.
        
        Args:
            test_perc (float): percentile of test set with respect to the total data.
            train_perc (float): percentile of training set with respect to the training & validation set. 
                                In case of Hold-Out Validation (use K = 1).
            K (int): number of splits of the training & validation set. In case of K-Fold Cross Validation.
        
        Returns:
            x_trains (list): list of training sets extracted from training & validation set (the list has one element per iteration).
            target_trains (list): list of targets corrisponding to the training set.
            x_vals (list): list of validation sets extracted from training & validation set (the list has one element per iteration).
            target_vals (list): list of targets corrisponding to the validation set.
            x_test (array): test set extracted from input data.
            target_test (array): targets corresponding to the test set.
        '''

        if (train_perc > 1 or train_perc < 0):
            raise ValueError(f"Invalid {train_perc}. Choose from 0 to 1")

        x_train_val, target_train_val, x_test, target_test = self.test_split(test_perc)

        x_trains = []
        target_trains = []
        x_vals = []
        target_vals = []

        num_samples = x_train_val.shape[0]
        fold_size = num_samples // K

        if K==1: # Hold-out Validation
            train_indices = np.arange(0, int(train_perc * num_samples))
            val_indices = np.setdiff1d(np.arange(num_samples), train_indices) # setdiff1d is the set difference 
                                                                              # between the first and the second set
            x_train, target_train = x_train_val[train_indices], target_train_val[train_indices] # Definition of training set 
                                                                                                # with matching targets
            x_val, target_val = x_train_val[val_indices], target_train_val[val_indices] # Definition of validation set 
                                                                                        # with matching targets
            x_vals.append(x_val)
            target_vals.append(target_val)
            x_trains.append(x_train)
            target_trains.append(target_train)

        else: # K-fold Cross Validation
            for k in range(K):
                # creating fold indices
                val_indices = np.arange(k * fold_size, (k + 1) * fold_size) # Creation of an array of indices with len = fold_size.
                                                                            # It contains the indices of the examples used in validation
                                                                            # set for the current fold.
                train_indices = np.setdiff1d(np.arange(num_samples), val_indices) # Creation of an array of indices with len = num_samples -
                                                                                  # len(val_indices). It contains the indices of all the examples
                                                                                  # except the ones used in the Validation set for this fold.
                                                                                  # It corresponds to the Training set for the current fold.
                x_train, target_train = x_train_val[train_indices], target_train_val[train_indices]
                x_val, target_val = x_train_val[val_indices], target_train_val[val_indices]

                # Shuffling
                new_train_indices = np.arange(x_train.shape[0])
                np.random.shuffle(new_train_indices)
                x_train = x_train[new_train_indices]
                target_train = target_train[new_train_indices]

                x_vals.append(x_val)
                target_vals.append(target_val)
                x_trains.append(x_train)
                target_trains.append(target_train)
        
        return x_trains, target_trains, x_vals, target_vals, x_test, target_test
    
# Unit test
np.random.seed(42)

x_tot = np.random.rand(10, 3)
target_tot = np.random.rand(10, 3)

test_perc = 0.2
K = 1

print(f"x tot: \n {x_tot}")

Data = DataProcessing(x_tot, target_tot)
x_trains, target_trains, x_vals, target_vals, x_test, target_test = Data.train_val_split(test_perc, K, 0.7)

print(f"x trains: \n {x_trains}")
print(f"x vals: \n {x_vals}")
print(f"x test: \n {x_test}")