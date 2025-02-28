from symbol import parameters
from init_utils import *
import numpy as np


def model(X, Y, learning_rate=0.01, num_iterations= 15000, print_cost=True, initialization="he"):
    """
    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (containing 0 for red dots; 1 for blue dots), of shape (1, number of examples)
    learning_rate -- learning rate for gradient descent
    num_iterations -- number of iterations to run gradient descent
    print_cost -- if True, print the cost every 1000 iterations
    initialization -- flag to choose which initialization to use ("zeros","random" or "he")

    Returns:
    parameters -- parameters learnt by the model
    """
    grads = {}
    costs = []
    m = X.shape[1] # number of examples
    layers_dims = [X.shape[0], 10, 5, 1]

    if initialization == "zeros":
        params = initial_parameters_zeros(layers_dims)
    elif initialization == "random":
        params = initial_parameters_random(layers_dims)
    elif initialization == "he":
        params = initial_parameters_he(layers_dims)

    for i in range(num_iterations):
        # Forward propagation:
        a3, cache = forward_propagation(X, params)

        # Lost
        cost = compute_loss(a3, Y)

        # Backward propagation:
        grads = backward_propagation(X, Y, cache)

        # Update parameters
        params = update_parameters(params, grads, learning_rate)

        if print_cost and i % 1000 == 0:
            print(f"Cost after iteration {i}: {cost}")
            costs.append(cost)
    return params
"""
Model	Train accuracy	Problem/Comment
3-layer NN with zeros initialization	50%	fails to break symmetry
3-layer NN with large random initialization	83%	too large weights
3-layer NN with He initialization	99%	recommended method
"""


def initial_parameters_zeros(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    params = {}
    L = len(layers_dims)

    for l in range(1, L):
        params['W' + str(l)] = np.zeros((layers_dims[l], layers_dims[l-1]))
        params['b' + str(l)] = np.zeros((layers_dims[l], 1))
    return params

def initial_parameters_random(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """

    np.random.seed(3)
    params = {}
    L = len(layers_dims)

    for l in range(1, L):
        params['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * 10
        params['b' + str(l)] = np.zeros((layers_dims[l], 1))
    return params

def initial_parameters_he(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """

    np.random.seed(3)
    params = {}
    L = len(layers_dims)

    for l in range(1, L):
        params['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2./layers_dims[l-1])
        params['b' + str(l)] = np.zeros((layers_dims[l], 1))
    return params

np.multi