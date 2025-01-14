import numpy as np
import matplotlib.pyplot as plt

class ModelAssessment:

    def __init__(
            self,
            x_retrain,
            target_retrain,
            x_test,
            target_test,
            epochs,
            batch_size,
            loss_func,
            d_loss_func,
            neural_network
    ):
        '''
        Class focused on the actual training and validation of the neural network.

        Args:
            INSERIRE
            epochs (int): number of iterations of the cycle forward propagation + backward propagation + weights update. 
            batch_size (int): batch size for training. If batch_size = 1, the neural network is trained using an online learning approach.
                              If batch_size != 1, the neural network is trained using a mini-batch learning approach with batches of size
                              batch_size.
            loss_func (func): loss function.
            d_loss_func (func): derivative of loss function.
            neural_network (NeuralNetwork): instance of the class NeuralNetwork.
        '''
        self.x_retrain = x_retrain
        self.target_retrain = target_retrain
        self.x_test = x_test
        self.target_test = target_test
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_func = loss_func
        self.d_loss_func = d_loss_func
        self.neural_network = neural_network


    def batch_generator( 
            self,
            xx,
            target
    ):
        '''
        Function that splits the training data into mini-batches.

        Args:
            INSERIRE

        Returns:
            x_batches (list): list of arrays of data that form the mini-batches.
            target_batches (list): list of arrays of labels corresponding to the data in x_batches.
        '''
        num_samples = xx.shape[0]
        x_batches = []
        target_batches = []
        for i in range(0, num_samples, self.batch_size):
            batch = xx[i : i + self.batch_size]
            target_batch = target[i : i + self.batch_size]
            x_batches.append(batch)
            target_batches.append(target_batch)
        
        return x_batches, target_batches
    
    def retrain_epoch(
            self,
            xx,
            target
    ):
        '''
        Function that computes the average training error through the training of the network for a single epoch.
        
        Args:
            INSERIRE
            
        Returns:
            train_error_epoch (float): average training error of one epoch. 
        '''
        batches, target_batches = self.batch_generator(xx, target)
        train_error_epoch = 0
        prediction = np.array([])

        for batch, target_batch in zip(batches, target_batches):
            #print(f'batch: \n {batch}')
            pred = self.neural_network.forward(batch)
            prediction = np.append(prediction, pred)
            #print(f'pred: \n {pred}')
            train_error_epoch += self.loss_func(target_batch, pred)
            d_loss = self.d_loss_func(target_batch, pred)
            self.neural_network.backward(d_loss)
            #print(f'd_loss: \n {d_loss}')

        train_error_epoch /= xx.shape[0]

        return train_error_epoch, prediction
    
    
    def test_epoch(
            self,
            xx,
            target
    ):
        '''
        Function that computes the average test error through the training of the network for a single epoch.        

        Returns:
            test_error_epoch (float): average test error of one epoch.
        '''
        pred = self.neural_network.forward(xx)
        test_error_epoch = self.loss_func(target, pred)/xx.shape[0]

        return test_error_epoch, pred

    
    def accuracy_curve(
            self,
            pred,
            target
    ): 
        #pred = self.neural_network.forward(xx)
        correct_classifications = 0
        
        for k in range(len(pred)):
            print(f'pred prima = {pred[k]}')
            pred[k] = 1 if pred[k] >= 0.5 else 0
            #print(f'pred dopo = {pred[k]}')
            if (pred[k] == target[k]):
                correct_classifications += 1

        accuracy = correct_classifications/len(pred)
        #print(f'corret class: {correct_classifications}')
        #print(f'len pred: {len(pred)} ')
        #print(f'accuracy: {accuracy}')

        return accuracy

    def plot_accuracy(
            self,
            accuracy_retrain,
            accuracy_test
    ):
        plt.plot(accuracy_retrain, label = 'Training Accuracy')
        plt.plot(accuracy_test, label = 'Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.legend()
        plt.savefig('figura1.png')
        plt.show()


    def retrain_test(
            self,
            accuracy_check = False           
    ):
        '''
        INSERIRE        
        '''
        retrain_error_tot = np.array([])
        test_error_tot = np.array([])

        accuracy_retrain_tot = np.array([])
        accuracy_test_tot = np.array([])

        for i in range(self.epochs):
            retrain_error_epoch, retrain_pred = self.retrain_epoch(self.x_retrain, self.target_retrain)
            retrain_error_tot = np.append(retrain_error_tot, retrain_error_epoch)

            test_error_epoch, test_pred = self.test_epoch(self.x_test, self.target_test)
            test_error_tot = np.append(test_error_tot, test_error_epoch)
            
            if accuracy_check:
                accuracy_retrain = self.accuracy_curve(retrain_pred, self.target_retrain)
                accuracy_test = self.accuracy_curve(test_pred, self.target_test)

                accuracy_retrain_tot = np.append(accuracy_retrain_tot, accuracy_retrain)
                accuracy_test_tot = np.append(accuracy_test_tot, accuracy_test)


            if ((i+1)%10 == 0):
                print(f'epoch {i+1}, retrain error {retrain_error_epoch}, test error {test_error_epoch}')


        if accuracy_check:
            self.plot_accuracy(accuracy_retrain_tot, accuracy_test_tot)
            print(f"Final training accuracy value: {accuracy_retrain_tot[-1]}")
            print(f"Final test accuracy value: {accuracy_test_tot[-1]}")
            print(f'lunghezza accuracy train {len(accuracy_retrain_tot)}')
            print(f'lunghezza accuracy test {len(accuracy_test_tot)}')

        return retrain_error_tot, test_error_tot
