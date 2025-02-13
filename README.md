# Machine Learning Project - Introduction and objectives

Repository about Machine Learning project: the aim is to build a multi layer perceptron from scratch, able to perform binary classification and to solve regression problems.
The neural network is trained using gradient descent algorithm combined with backpropagation.
Once the perceptron is built, the focus shifted on improving it: best fitting hyperparameters are found through grid search supported by successive halvings algorithm; elastic regularization stabilizes the training and Adam and Nesterov Accelerated Gradient optimizers are added for the learning algorithm.
The implementation of the neural network has been made from scratch in Python, using Numpy, Matplotlib and Pandas as support libraries.

# BandieraMarlia_Bini_Montanelli directory material:

## Codes and Class explanation for the building and the evaluation of the neural network, including model selection, model assessment and grid search for the hyperparameters

- 'RegularizationOptimizationClass.py' implements the classes Regularization and Optimization: the former implements the regularization term, the latter implements the learning rule with the optimizer (Adam, NAG or no optimizer) chosen and the previous regularization term, updating the weights and returning the loss gradient for the hidden layers used in backpropagation; 

- 'Functions.py' implements the dictionary of the activation functions used in the layers and the loss functions;

- 'LayerClass.py' implements class Layer: the basic building brick of the network. 
Layers are constituted with their input dimension, output dimension and activation. The weights are initialized extracting randomly from a uniform distribution depending from the fan-in and the biases are initially set to zero. A forward and backward methods are also implemented;

- 'NeuralNetworkClass.py' implements class NeuralNetwork, which associates the layer configuration to a regulizer and an optimizer respectively passed from classes Regularization and Optimization. Here, it is possible to choose the type of regularization and optimization and to tune their parameters;

- 'DataProcessingClass.py' implements the class DataProcessing, designed to divide data in training+validation set (80%) and test set (20%). Afterwards, the training+validation data is set to be employed as hold-out validation or a k-fold cv;

- 'ModelSelectionClass.py' implements Model Selection, a class that allows the training algorithm to be performed on the previous setted data with batch method (online/mini-batch/batch), returning the training and validation loss for each epoch.

- 'LossControlClass.py' implements the class LossControl, useful to evaluate the goodnees of the error functions. Here three stopping checks are realized, based on early stopping, smoothness and overfitting;

- 'GeneralGridBuilding.py' implements the building of the grid (in a dictionary form) in the hyperparameters space in order to allow the grid search;

- 'SuccessiveHalvings.py' performs successive halvings grid search on the hyperparameters. It takes the combinations of the dictionary of the grid building in order to form the layer configuration, the regulizer configuration and the optimizer configuration to give as parameters to the NeuralNetwork instance. Then it builds the actual neural networks. 
In order to perform the training algorithm, it calculates the combination of neural networks with the possible batch sizes. Finally, it returns the best computed configurations and it writes them also in a .txt file.
It's used to find the most robust architectures of the neural network;

- 'GridSearchClass.py' performs a final, fine, total exploration of the grid  in order to fix the other hyperparameters, such as the learning rate, the batch size, the lambda term of the regularization, the momentum term alpha and the best optimizer.
It saves the plots of the learning curve for each configuration tested. 

## Codes explanation for Monk dataset

The basic assessment of the neural network is tested on Monk's dataset with some poor configuration (1 hidden layer, few units, no optimizer).

- 'MonkDataProcessing.py' extracts data from .csv files containing Monk datasets, it splits the Monk data into training set and test set converting datas with one hot encoding, finally it converts pandas dataframes in numpy arrays;

- 'Monk1.py' solves the Monk problem for Monk1 dataset, showing the learning curve and the accuracy plot;

- 'Monk2.py' solves the Monk problem for Monk2 dataset, showing the learning curve and the accuracy plot;

- 'Monk3.py' solves the Monk problem for Monk3 dataset, showing the learning curve and the accuracy plot;

- 'Monk3_NAG_none.py' compares the Monk3 problem using NAG optimizer vs no optimizer, showing the learning curve and the accuracy plot of both configurations;

- 'Monk3_adam_none.py' compares the Monk3 problem using adam optimizer vs no optimizer, showing the learning curve and the accuracy plot of both configurations.

## Codes explanation for CUP dataset

After its implementation, the neural network is improved to solve a regression problem with a dataset (called CUP dataset) provided by the teacher of Machine Learning course. The dataset consists in 250 examples with 15 features each one. The neural network should be able to solve the regression problem distinguishing the 3 important features (x, y, z coordinates) from the other 12 noisy features. Additionally, a blind test set is provided, where the target values are held by the teacher. The results are compared with those of other groups participating in the same project, and the final outcomes are ranked based on the lowest error on the blind test.

- 'CUPDataProcessing.py' extracts data from .csv files containing CUP dataset; splits the CUP labelled data into training set, validation set and test set using DataProcessing class; processes the CUP unlabelled data (blind data); converts pandas dataframes in numpy arrays.

- 'CUPUnitTest.py' performs the training and validation for a chosen configuration with CUP dataset, showing the learning curve;

- 'CUPUnitTest_ModelAssessment.py' performs the model assessment for the best configuration of hyperparameters for CUP, showing the retraining error and test error vs epochs curve. Additionally, it returns the predicted target values on the blind test writing in the file 'BG_peppers_ML-CUP24-TS.csv'.

## Other info

- 'BG_peppers_abstract.txt' contains a short summary of the characteristics of the chosen model and its training, validation and test error on CUP dataset;

- 'BandieraMarlia_Bini_Montanelli.pdf': presentation slides about the project. 

# Dataset Material

- 'monk+s+problems' directory: monks problem dataset;

- 'ML-CUP24-TR.csv': CUP training dataset;

- 'ML-CUP24-TS.csv': CUP tblid test dataset.


