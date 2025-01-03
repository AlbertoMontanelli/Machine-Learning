import numpy as np

# Definition of activation functions and their derivative
def sigmoid(net):
    """ 
    Sigmoid function.

    Args:
        net (array): array computed as input * weights + bias.

    Returns:
        new array (array): result of Sigmoid function.
    """  
    return 1 / (1 + np.exp(-net))


def d_sigmoid(net):
    """ 
    Derivative of Sigmoid function.

    Args:
        net (array): array computed as input * weights + bias.

    Returns:
        new array (array): result of the derivative of Sigmoid function.
    """ 
    return np.exp(-net) / (1 + np.exp(-net))**2


def tanh(net):
    """ 
    Tanh function.

    Args:
        net (array): array computed as input * weights + bias.

    Returns:
        new array (array): result of Tanh function.
    """ 
    return np.tanh(net)


def d_tanh(net):
    """ 
    Derivative of Tanh function.

    Args:
        net (array): array computed as input * weights + bias.

    Returns:
        new array (array): result of the derivative of Tanh function.
    """ 
    return 1 - (np.tanh(net))**2


def softplus(net):
    """ 
    Softplus function.

    Args:
        net (array): array computed as input * weights + bias.

    Returns:
        new array (array): result of Softplus function.
    """
    return np.log(1 + np.exp(net))


def d_softplus(net):
    """ 
    Derivative of Softplus function.

    Args:
        net (array): array computed as input * weights + bias.

    Returns:
        new array (array): result of the derivative of Softplus function.
    """ 
    return np.exp(net) / (1 + np.exp(net))


def linear(net):
    """ 
    Linear function.

    Args:
        net (array): array computed as input * weights + bias.

    Returns:
        new array (array): result of Linear function.
    """ 
    return net


def d_linear(net):
    """ 
    Derivative of Linear function.

    Args:
        net (array): array computed as input * weights + bias.

    Returns:
        new array (array): result of the derivative of Linear function.
    """ 
    return np.ones_like(net)


def ReLU(net):
    """ 
    ReLU function.

    Args:
        net (array): array computed as input * weights + bias.

    Returns:
        new array (array): result of ReLU function.
    """ 
    return np.maximum(net, np.zeros_like(net))


def d_ReLU(net):
    """ 
    Derivative of ReLU function.

    Args:
        net (array): array computed as input * weights + bias.

    Returns:
        new array (array): result of the derivative of ReLU function.
    """ 
    return np.ones_like(net) if(net>=0) else np.zeros_like(net)


def leaky_relu(net, alpha):
    """ 
    Leaky ReLU function.

    Args:
        net (array): array computed as input * weights + bias.

    Returns:
        new array (array): result of Leaky ReLU function.
    """ 
    return np.maximum(net, alpha*net)


def d_leaky_relu(net, alpha):
    """ 
    Derivative of Leaky ReLU function.

    Args:
        net (array): array computed as input * weights + bias.

    Returns:
        new array (array): result of the derivative of Leaky ReLU function.
    """ 
    return np.ones_like(net) if(net>=0) else alpha


def ELU(net):
    """ 
    ELU function.

    Args:
        net (array): array computed as input * weights + bias.

    Returns:
        new array (array): result of ELU function.
    """ 
    return net if(net>=0) else np.exp(net)-1


def d_ELU(net):
    """ 
    Derivative of ELU function.

    Args:
        net (array): array computed as input * weights + bias.

    Returns:
        new array (array): result of ELU function.
    """ 
    return np.ones_like(net) if(net>=0) else np.exp(net)


# np.vectorize returns an object that acts like pyfunc, but takes arrays as input
d_ReLU = np.vectorize(d_ReLU)
d_leaky_relu = np.vectorize(d_leaky_relu)
ELU = np.vectorize(ELU)
d_ELU = np.vectorize(d_ELU)

# Dictionary for the activation functions
activation_functions = {
    "sigmoid": sigmoid,
    "tanh": tanh,
    "softplus": softplus,
    "linear": linear,
    "ReLU": ReLU,
    "leaky_relu": leaky_relu,
    "ELU": ELU,
}

# Dictionary for the derivative of activation functions
d_activation_functions = {
    "d_sigmoid": d_sigmoid,
    "d_tanh": d_tanh,
    "d_softplus": d_softplus,
    "d_linear": d_linear,
    "d_ReLU": d_ReLU,
    "d_leaky_relu": d_leaky_relu,
    "d_ELU": d_ELU,
}


# Definition of Loss functions and their derivative
def mean_squared_error(y_true, y_pred):
    """ 
    Mean Squared Error function.

    Args:
        y_true (array): targets that represent data labels provided as input.
        y_pred (array): output of the last layer.

    Returns:
        new float (float): result of Mean Squared Error.
    """ 
    return np.sum((y_true - y_pred)**2)


def d_mean_squared_error(y_true, y_pred):
    """ 
    Derivative of Mean Squared Error function.

    Args:
        y_true (array): targets that represent data labels provided as input.
        y_pred (array): output of the last layer.

    Returns:
        new array (array): result of the derivative of Mean Squared Error.
    """ 
    return - 2 * (y_true - y_pred)


def mean_euclidean_error(y_true, y_pred):
    """ 
    Mean Euclidean Error function.

    Args:
        y_true (array): targets that represent data labels provided as input.
        y_pred (array): output of the last layer.

    Returns:
        new float (float): result of Mean Euclidean Error.
    """ 
    return np.sqrt(np.sum((y_true - y_pred)**2))


def d_mean_euclidean_error(y_true, y_pred):
    """ 
    Derivative of Mean Euclidean Error function.

    Args:
        y_true (array): targets that represent data labels provided as input.
        y_pred (array): output of the last layer.

    Returns:
        new array (array): result of the derivative of Mean Euclidean Error.
    """ 
    return - (y_true - y_pred) / np.sqrt(np.sum((y_true - y_pred)**2))


def huber_loss(y_true, y_pred, delta):
    """ 
    Huber Loss function.

    Args:
        y_true (array): targets that represent data labels provided as input.
        y_pred (array): output of the last layer.
        delta (float): parameter that defines the threshold for the transition between two error regimes.

    Returns:
        new array (array): result of Huber Loss.
    """ 
    return 0.5 * (y_true - y_pred)**2 if(np.abs(y_true-y_pred)<=delta) else delta * np.abs(y_true - y_pred) - 0.5 * delta**2


def d_huber_loss(y_true, y_pred, delta):
    """ 
    Derivative of Huber Loss function.

    Args:
        y_true (array): targets that represent data labels provided as input.
        y_pred (array): output of the last layer.
        delta (float): parameter that defines the threshold for the transition between two error regimes.

    Returns:
        new array (array): result of the derivative of Huber Loss.
    """ 
    return - (y_true - y_pred) if(np.abs(y_true-y_pred)<=delta) else - delta * np.sign(y_true-y_pred)


# Dictionary for the loss functions
loss_functions = {
    "mse": mean_squared_error,
    "mee": mean_euclidean_error,
    "huber": huber_loss,
}

# Dictionary for the derivative of loss functions
d_loss_functions = {
    "d_mse": d_mean_squared_error,
    "d_mee": d_mean_euclidean_error,
    "d_huber": d_huber_loss,
}