import numpy as np

np.random.seed(12)

class DataProcessing:

    def __init__(self, x_tot, target_tot, test_perc = 0.2, K = 1, train_perc = 0.75):
        '''
        Class focused on processing the data into Training Set, Validation Set and Test Set

        Args:
            x_tot (array): total data given as input.
            target_tot (array): total data labels given as input.
            test_perc (float): percentile of test set with respect to the total data.
            K (int): number of splits of the training + validation set. In case of Hold-Out Validation K = 1. 
                     K > 1 is the number of folds used in K-Fold Cross Validation.
            train_perc (float): percentile of training set with respect to the training + validation set. 
        '''

        self.x_tot = x_tot
        self.target_tot = target_tot
        self.test_perc = test_perc
        self.K = K
        self.train_perc = train_perc

        if (K < 1):
            raise ValueError(f"Invalid value for the number of splits of the training + validation set K = {K}. Choose an integer >= 1.")

        num_samples = self.x_tot.shape[0]
        indices = np.arange(num_samples)
        np.random.shuffle(indices) 
        self.x_tot = self.x_tot[indices]
        self.target_tot = self.target_tot[indices]

        self.x_train_val, self.target_train_val, self.x_test, self.target_test = self.test_split()
        self.x_trains, self.target_trains, self.x_vals, self.target_vals = self.train_val_split()

        # If no test set is needed, test data remains None
        if self.x_test is None and self.target_test is None:
            print("No test split performed. Entire dataset used for training and validation.")


    def test_split(self):
        '''
        Function that splits the input data into two sets: training + validation set, test set.
        
        Returns:
            x_train_val (array): training + validation set extracted from input data.
            target_train_val (array): training + validation set labels.
            x_test (array): test set extracted from input data.
            target_test (array): test set for input data labels.
        '''

        if not (0 <= self.test_perc <= 1):
            raise ValueError(f"Invalid test set percentile: {self.test_perc}. Choose from 0 to 1")
        
        num_samples = self.x_tot.shape[0] # the total number of the examples in input = the number of rows in the x_tot matrix

        if self.test_perc == 0:
            return self.x_tot, self.target_tot, None, None

        test_size = int(num_samples * self.test_perc) # the number of the examples in the test set
        
        indices = np.arange(num_samples) # it creates an array of indices arranged from 0 to num_samples 
        test_indices = indices[:test_size]
        train_val_indices = indices[test_size:]

        x_test = self.x_tot[test_indices]
        target_test = self.target_tot[test_indices]
        x_train_val = self.x_tot[train_val_indices]
        target_train_val = self.target_tot[train_val_indices]

        return x_train_val, target_train_val, x_test, target_test
    
    
    def train_val_split(self):
        '''
        Function that splits the training & validation set into two sets: training set and validation set.
        
        Returns:
            x_trains (list): list of training sets extracted from training + validation set (the list has one element per iteration).
            target_trains (list): list of targets corrisponding to the training set.
            x_vals (list): list of validation sets extracted from training + validation set (the list has one element per iteration).
            target_vals (list): list of targets corrisponding to the validation set. 
        '''
        
        if not (0 <= self.train_perc <= 1):
            raise ValueError(f"Invalid traininig set percentile: {self.train_perc}. Choose from 0 to 1")

        num_samples = self.x_train_val.shape[0]
        indices = np.arange(num_samples)

        if self.K == 1: # Hold-Out Validation
            train_size = int(self.train_perc * num_samples)
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]

            x_trains = [self.x_train_val[train_indices]]
            target_trains = [self.target_train_val[train_indices]]
            x_vals = [self.x_train_val[val_indices]]
            target_vals = [self.target_train_val[val_indices]]

        else: # K-fold Cross Validation
            fold_size = num_samples // self.K
            x_trains, target_trains, x_vals, target_vals = [], [], [], []

            for k in range(self.K):
                # creating fold indices
                val_indices = np.arange(k * fold_size, (k + 1) * fold_size) # creation of an array of indices with len = fold_size.
                                                                            # It contains the indices of the examples used in validation set for
                                                                            # this fold.
                train_indices = np.setdiff1d(np.arange(num_samples), val_indices) # creation of an array of indices with len = num_samples -
                                                                                # len(val_indices). It contains the indices of all the examples
                                                                                # but the ones used in the validation set for this fold.
                                                                                # It corresponds to the training set for the current fold.
                x_train, target_train = self.x_train_val[train_indices], self.target_train_val[train_indices]
                x_val, target_val = self.x_train_val[val_indices], self.target_train_val[val_indices]

                # Shuffling
                new_train_indices = np.arange(x_train.shape[0])
                np.random.shuffle(new_train_indices)
                x_train = x_train[new_train_indices]
                target_train = target_train[new_train_indices]

                x_vals.append(x_val)
                target_vals.append(target_val)
                x_trains.append(x_train)
                target_trains.append(target_train)
        
        return x_trains, target_trains, x_vals, target_vals