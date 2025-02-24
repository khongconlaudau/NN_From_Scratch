from symbol import parameters

import copy
import numpy as np
from dnn_utils import *
"""
For example, the `layer_dims` for last week's Planar Data classification model would have been [2,4,1]: There were two inputs, one hidden layer with 4 hidden units, and an output layer with 1 output unit. This means `W1`'s shape was (4,2), `b1` was (4,1), `W2` was (1,4) and `b2` was (1,1). Now you will generalize this to $L$ layers!
"""

def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    np.random.seed(3)
    parameters = {}
    for l in range(1, len(layer_dims)):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        """
        np.random.rand: Generates random numbers from a uniform distribution between 0 and 1.
        
        np.random.randn: Generates random numbers from a standard normal distribution (also called a Gaussian distribution) with:
        Values can be both positive and negative, unlike rand(), which only returns values between 0 and 1.
        """
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

"""
𝑍[𝑙]=𝑊[𝑙]𝐴[𝑙−1]+𝑏[𝑙]
"""
def linear_foward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    Z = np.dot(W, A) + b

    cache = (A, W, b)

    return Z, cache



def linear_activation_foward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """

    Z, linear_cache = linear_foward(A_prev, W, b)

    if activation == "relu":
        A, activation_cache = relu(Z)
    elif activation == "sigmoid":
        A, activation_cache = sigmoid(Z)


    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_foward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- activation value from the output (last) layer
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L of them, indexed from 0 to L-1)
    """

    caches = []

    A = X
    L = len(parameters) // 2 # number of layers in the neural network
    for l in range(1, L):
        A_prev = A

        A, cache = linear_activation_foward(A_prev,
                                            parameters['W' + str(l)],
                                            parameters['b' + str(l)],
                                            activation="relu")
        caches.append(cache)


    AL, cache = linear_activation_foward(A,
                                         parameters['W' + str(L)],
                                         parameters['b' + str(L)],
                                         activation="sigmoid")
    caches.append(cache)

    return AL, caches



"""
−1𝑚∑𝑖=1𝑚(𝑦(𝑖)log(𝑎[𝐿](𝑖))+(1−𝑦(𝑖))log(1−𝑎[𝐿](𝑖)))
"""

def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """

    m = Y.shape[1]  # number of examples

    cost = - (1/m) * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))

    cost = np.squeeze(cost)   # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

    return cost


"""
𝑑𝑊[𝑙]=∂∂𝑊[𝑙]=1/𝑚 * 𝑑𝑍[𝑙]𝐴[𝑙−1]𝑇
𝑑𝑏[𝑙]=∂∂𝑏[𝑙]=1𝑚∑𝑖=1/𝑚 * 𝑑𝑍[𝑙](𝑖)
𝑑𝐴[𝑙−1]=∂∂𝐴[𝑙−1]=𝑊[𝑙]𝑇𝑑𝑍[𝑙]
"""

def linear_backward(dZ, cache):
    """
        Implement the linear portion of backward propagation for a single layer (layer l)

        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """

    A_prev, W, b = cache
    m = A_prev.shape[1] # number of examples

    dW = 1 / m * np.dot(dZ, A_prev.T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """

    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = relu_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

"""
dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # derivative of cost with respect to AL
"""
def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """

    grads = {}
    L = len(caches)
    m = Y.shape[1]
    Y = Y.reshape(AL.shape) # make sure prediction and y label have the same shape

    # Initializing the backpropagation
    dAL = - (np.divide(Y, Al) - np.divide(1 - Y, 1 - AL))

    # Last Layer
    current_cache =  caches[L - 1]
    dA_prev, dW, db = linear_activation_backward(dAl, current_cache, activation="sigmoid")
    grads["dA" + str(L - 1)] = dA_prev
    grads["dW" + str(L)] = dW
    grads["db" + str(L)] = db

    # Loop from l=L-2 to l=0
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_curr = grads["dA" + str(l + 1)]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dA_curr, current_cache, activation="relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    params -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """
    params = copy.deepcopy(parameters)
    L = len(params) // 2

    for l in range(L):
        params["W" + str(l + 1)] = params["W" + str(l + 1)]  - learning_rate * grads["dW" + str(l + 1)]
        params["b" + str(l + 1)] = params["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return params



