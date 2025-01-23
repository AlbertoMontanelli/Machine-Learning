import numpy as np

class ModelSelection:

    def __init__(
            self,
            data_splitter,
            epochs,
            batch_size,
            loss_func,
            d_loss_func,
            neural_network,
            loss_control
    ):
        '''
        Class focused on the actual training and validation of the neural network.

        Args:
            data_splitter (DataProcessing): instance of the class DataProcessing.
                Returns:
                    x_train (array): data through which the neural network will be trained.
                    target_train (array): targets of x_train.
                    x_val (array): data through which the neural network is validated.
                    target_val (array): targets of x_val.
            epochs (int): number of iterations of the training cycle (forward propagation + backward propagation). 
            batch_size (int): batch size for training. If batch_size = 1, the neural network is trained using an online learning approach.
                              If batch_size != 1, the neural network is trained using a mini-batch learning approach with batches of size
                              batch_size.
            loss_func (func): loss function.
            d_loss_func (func): derivative of the loss function.
            neural_network (NeuralNetwork): instance of the class NeuralNetwork.
            RISCRIVERE loss_control (EarlyStoppingClass): RISCRIVEREE
        '''
        self.data_splitter = data_splitter
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_func = loss_func
        self.d_loss_func = d_loss_func
        self.loss_control = loss_control
        self.neural_network = neural_network


    def batch_generator( 
            self,
            x_train,
            target_train
    ):
        '''
        Function that splits the training data into mini-batches.

        Args:
            x_train (array): one array of the list of arrays of self.data_splitter.x_trains.
            target_train (array): targets corresponding to self.x_train.

        Returns:
            x_batches (list): list of arrays of data that form the mini-batches.
            target_batches (list): list of arrays of labels corresponding to the data in x_batches.
        '''
        num_samples = x_train.shape[0]
        if self.batch_size > num_samples:
            raise ValueError(f'Invalid batch size {self.batch_size}. Must be smaller than number of examples {num_samples}')
        x_batches = []
        target_batches = []
        for i in range(0, num_samples, self.batch_size):
            batch = x_train[i : i + self.batch_size]
            target_batch = target_train[i : i + self.batch_size]
            x_batches.append(batch)
            target_batches.append(target_batch)
        
        return x_batches, target_batches


    def train_epoch(
            self,
            x_train,
            target_train
    ):
        '''
        Function that computes the average training error through the training of the network for a single epoch.
        
        Args:
            x_train (array): one array of the list of arrays of self.data_splitter.x_trains.
            target_train (array): targets corresponding to x_train.
            
        Returns:
            train_error_epoch (float): average training error of one epoch.
        '''
        batches, target_batches = self.batch_generator(x_train, target_train)
        train_error_epoch = 0
        
        for batch, target_batch in zip(batches, target_batches):
            pred = self.neural_network.forward(batch)
            train_error_epoch += self.loss_func(target_batch, pred)
            d_loss = self.d_loss_func(target_batch, pred)
            self.neural_network.backward(d_loss)

        train_error_epoch /= x_train.shape[0]

        return train_error_epoch
    
    
    def train_val(
            self,
            x_val,
            target_val
    ):
        '''
        Function that computes the average validation error through the training of the network for a single epoch.
        
        Args:
            x_val (array): one array of the list of arrays of self.data_splitter.x_vals.
            target_val (array): targets corresponding to x_val.
            
        Returns:
            val_error_epoch (float): average training error of one epoch.
        '''
        pred = self.neural_network.forward(x_val)
        val_error_epoch = self.loss_func(target_val, pred)/x_val.shape[0]

        return val_error_epoch

    def loss_control_avg(
            self, 
            train_error,
            val_error,
            overfitting,
            early_stopping,
            smoothness
    ):
        '''
        doc
        '''
        epochs = len(train_error)
        stop_epoch = epochs
        for epoch in range(epochs):
            if overfitting:
                overfitting_check = self.loss_control.overfitting_check(epoch, train_error, val_error)
                if overfitting_check:
                    print(f"Overfitting at epoch {epoch}")
                    stop_epoch = epoch - self.loss_control.overfitting_patience  # Registra l'epoca di stop (inclusiva) 
                    break

            if early_stopping:
                early_check = self.loss_control.stopping_check(epoch, val_error)
                if early_check:
                    print(f"Early stopping at epoch {epoch}")
                    stop_epoch = epoch - self.loss_control.stopping_patience  # Registra l'epoca di stop (inclusiva)
                    break

            if smoothness:
                smoothness_check_train = self.loss_control.smoothness_check(epoch, train_error)
                
                if (smoothness_check_train == False):
                    #print(f"Loss function not smooth for fold {fold_idx+1}")
                    smoothness_check = False
                else:
                    smoothness_check = True

        return smoothness_check, stop_epoch                 


    def train_fold(
            self,
            early_stopping = False,
            smoothness = False,
            overfitting = False
    ):
        '''
        Function that computes training and validation error averaged on the number of folds for each epoch.

        Returns:
            train_error_tot (array): training error for each epoch averaged on the number of folds.
            val_error_tot (array): validation error for each epoch averaged on the number of folds.
        
        '''
        train_error_tot = []
        val_error_tot = []

        for fold_idx, (x_train, target_train, x_val, target_val) in enumerate(
            zip(
                self.data_splitter.x_trains,
                self.data_splitter.target_trains,
                self.data_splitter.x_vals,
                self.data_splitter.target_vals,
            )
        ):
            if self.neural_network.grid_search == False:
                print(f'fold n: {fold_idx + 1}')

            train_error = []
            val_error = []

            for i in range(self.epochs):
                train_error_epoch = self.train_epoch(x_train, target_train)
                val_error_epoch = self.train_val(x_val, target_val)

                train_error.append(train_error_epoch)
                val_error.append(val_error_epoch)          

                if self.neural_network.grid_search == False:
                    if ((i + 1) % 10 == 0):
                        print(f'epoch {i+1}, train error {train_error_epoch}, val error {val_error_epoch}')
                
            train_error_tot.append(train_error)
            val_error_tot.append(val_error)

            self.neural_network.reinitialize_net_and_optimizers()


        # Media sui fold
        train_error_avg = np.mean(train_error_tot, axis=0)
        val_error_avg = np.mean(val_error_tot, axis=0)

        if smoothness or early_stopping or overfitting:
            print('entra?')
            smoothness_outcome, stop_epoch = self.loss_control_avg(train_error_avg, val_error_avg, overfitting, early_stopping, smoothness)
            train_error_avg = train_error_avg[:stop_epoch]
            val_error_avg = val_error_avg[:stop_epoch]
            
        if self.neural_network.grid_search == False:
            print(f'last val error: \n {val_error_avg[-1]}')
            print(f'last train error: \n {train_error_avg[-1]}')

        if smoothness:
            return train_error_avg, val_error_avg, smoothness_outcome
        else:
            return train_error_avg, val_error_avg

    '''
    def train_fold(
            self,
            early_stopping = False,
            smoothness = False,
            overfitting = False
    ):

        train_error_tot = []
        val_error_tot = []

        # Indici per monitorare eventuale early stopping
        stop_epochs = np.zeros(self.data_splitter.K, dtype=int)
        smoothness_fold = np.zeros(self.data_splitter.K, dtype=bool)

        for fold_idx, (x_train, target_train, x_val, target_val) in enumerate(
            zip(
                self.data_splitter.x_trains,
                self.data_splitter.target_trains,
                self.data_splitter.x_vals,
                self.data_splitter.target_vals,
            )
        ):
            if self.neural_network.grid_search == False:
                print(f'fold n: {fold_idx + 1}')
            train_error = []
            val_error = []

            for i in range(self.epochs):
                train_error_epoch = self.train_epoch(x_train, target_train)
                val_error_epoch = self.train_val(x_val, target_val)

                train_error.append(train_error_epoch)
                val_error.append(val_error_epoch)


                if overfitting:
                    overfitting_check = self.loss_control.overfitting_check(i, train_error, val_error)
                    if overfitting_check:
                        print(f"Overfitting at epoch {i} for fold {fold_idx + 1}")
                        stop_epochs[fold_idx] = i + 1  # Registra l'epoca di stop (inclusiva)
                        break
                    else:
                        stop_epochs[fold_idx] = self.epochs 

                if early_stopping:
                    early_check = self.loss_control.stopping_check(i, val_error)
                    if early_check:
                        print(f"Early stopping at epoch {i} for fold {fold_idx + 1}")
                        stop_epochs[fold_idx] = i + 1  # Registra l'epoca di stop (inclusiva)
                        break
                    else:
                        stop_epochs[fold_idx] = self.epochs  # Se non si interrompe, registra il massimo delle epoche

                if smoothness:
                    smoothness_check_train = self.loss_control.smoothness_check(i, train_error)
                    # smoothness_check_val = self.loss_control.smoothness_check(i, val_error)
                    
                    if (smoothness_check_train == False):
                        #print(f"Loss function not smooth for fold {fold_idx+1}")
                        smoothness_fold[fold_idx] = False

                    else:
                        smoothness_fold[fold_idx] = True                   

                
                if self.neural_network.grid_search == False:
                    if ((i + 1) % 30 == 0):
                        print(f'epoch {i+1}, train error {train_error_epoch}, val error {val_error_epoch}')
                


            train_error_tot.append(train_error)
            val_error_tot.append(val_error)

            if early_stopping:
                self.loss_control.stop_count = 0

            if overfitting:
                self.loss_control.overfitting_count = 0

            self.neural_network.reinitialize_net_and_optimizers()

        # Epoca massima su tutti i fold
        max_epoch = np.max(stop_epochs)

        # Normalizza le lunghezze degli array di errori dei fold
        for fold_idx in range(self.data_splitter.K):
            train_error_tot[fold_idx] += [train_error_tot[fold_idx][-1]] * (max_epoch - len(train_error_tot[fold_idx]))
            val_error_tot[fold_idx] += [val_error_tot[fold_idx][-1]] * (max_epoch - len(val_error_tot[fold_idx]))

        

        # Media sui fold
        train_error_avg = np.mean(train_error_tot, axis=0)
        val_error_avg = np.mean(val_error_tot, axis=0)

        if self.neural_network.grid_search == False:
            print(f'last val error: \n {val_error_avg[-1]}')
            print(f'last train error: \n {train_error_avg[-1]}')

        smoothness_outcome = all(smoothness_fold) # if there is one False in the smoothness of the folds, smoothness_outcome is False 
        
        if smoothness:
            return train_error_avg, val_error_avg, smoothness_outcome
        else:
            return train_error_avg, val_error_avg
    '''

'''
Unit test for batches

np.random.seed(42)
x_tot = np.random.rand(10, 3)
print(f'x_tot pre-shuffle \n {x_tot}')
target_tot = np.random.rand(10, 3)
K = 3

data_split = DataProcessing(x_tot, target_tot, 0, K)
print(f'x_tot \n {data_split.x_trains}')
train_val = TrainingValidation(data_split, 100, 2)
for xx, target in zip(data_split.x_trains, data_split.target_trains):
    batches, target_batches = train_val.batch_generator(xx, target)
    print(f'batches \n {batches}')

'''

'''
Unit test for the training

for xx, target in zip(data_split.x_trains, data_split.target_trains):
    print(f"\n INIZIO FOLD \n")

    batches, target_batches = train_val.batch_generator(xx, target)
    for i in range(epochs):
        loss = 0
        for batch, target_batch in zip(batches, target_batches):
            pred = nn.forward(batch)
            loss += loss_functions['mse'](target_batch, pred)
            d_loss = d_loss_functions['d_mse'](target_batch, pred)
            
            nn.backward(d_loss)
        loss /= xx.shape[0]
        print(f"loss: \n {loss}")
    
    
    nn.reinitialize_net_and_optimizers()
'''