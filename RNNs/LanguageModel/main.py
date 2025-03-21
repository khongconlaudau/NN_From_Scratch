import numpy as np
import copy

from RNNs.LanguageModel.utils import rnn_forward, rnn_backward, update_parameters, initialize_parameters, \
    get_initial_loss
from RNNs.rnn_utils import softmax


def clip(gradients, maxValue):
    '''
    Clips the gradients' values between minimum and maximum.

    Arguments:
    gradients -- a dictionary containing the gradients "dWaa", "dWax", "dWya", "db", "dby"
    maxValue -- everything above this number is set to this number, and everything less than -maxValue is set to -maxValue

    Returns:
    gradients -- a dictionary with the clipped gradients.
    '''

    gradients = copy.deepcopy(gradients)

    for key in gradients:
        np.clip(gradients[key], -maxValue, maxValue, out=gradients[key])
    return gradients

def sample(parameters, char_to_idx, seed):
    """
    Sample a sequence of characters according to a sequence of probability distributions output of the RNN

    Arguments:
    parameters -- Python dictionary containing the parameters Waa, Wax, Wya, by, and b.
    char_to_ix -- Python dictionary mapping each character to an index.
    seed -- Used for grading purposes. Do not worry about it.

    Returns:
    indices -- A list of length n containing the indices of the sampled characters.
    """
    # Retrieve params from dic
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[0]

    # create tensor for x , a
    x = np.zeros((vocab_size, 1))
    a_prev = np.zeros((n_a, 1))

    indices = []
    idx = -1
    counter = 0
    newline_idx = char_to_idx['\n']

    while idx != newline_idx and counter != 50:
        a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
        z = np.dot(Wya, a) + by
        y = softmax(z)

        np.random.seed(counter + seed)
        idx = np.random.choice(range(vocab_size), p=y.ravel())

        indices.append(idx)
        # Reset x and take the prediction for next_x
        x = np.zeros((vocab_size, 1))
        x[idx] = 1

        a_prev = a

        seed += 1
        counter += 1
    if counter == 50:
        indices.append(newline_idx)
    return indices

def optimize(X, Y, a_prev, parameters, learning_rate=0.01 ):
    """
    Execute one step of the optimization to train the model.

    Arguments:
    X -- list of integers, where each integer is a number that maps to a character in the vocabulary.
    Y -- list of integers, exactly the same as X but shifted one index to the left.
    a_prev -- previous hidden state.
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        b --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    learning_rate -- learning rate for the model.

    Returns:
    loss -- value of the loss function (cross-entropy)
    gradients -- python dictionary containing:
                        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                        dWya -- Gradients of hidden-to-output weights, of shape (n_y, n_a)
                        db -- Gradients of bias vector, of shape (n_a, 1)
                        dby -- Gradients of output bias vector, of shape (n_y, 1)
    a[len(X)-1] -- the last hidden state, of shape (n_a, 1)
    """
    loss, cache = rnn_forward(X, Y, a_prev, parameters)
    gradients, a = rnn_backward(X, Y, parameters, cache)
    gradients = clip(gradients, 5)
    parameters = update_parameters(parameters, gradients, learning_rate)

    return loss, gradients,a[len(X)-1]

def model(data_x, ix_to_char, char_to_ix, num_iterations=35000, n_a = 50, dino_names = 7, vocab_size=27):
    """
    Trains the model and generates dinosaur names.

    Arguments:
    data_x -- text corpus, divided in words
    ix_to_char -- dictionary that maps the index to a character
    char_to_ix -- dictionary that maps a character to an index
    num_iterations -- number of iterations to train the model for
    n_a -- number of units of the RNN cell
    dino_names -- number of dinosaur names you want to sample at each iteration.
    vocab_size -- number of unique characters found in the text (size of the vocabulary)

    Returns:
    parameters -- learned parameters
    """

    n_x, n_y = vocab_size, vocab_size

    parameters = initialize_parameters(n_a, n_x, n_y)
    loss=  get_initial_loss(vocab_size, dino_names)

    examples = [x.strip() for x in data_x]
    np.random.seed(0)
    np.random.shuffle(examples)

    a_prev= np.zeros((n_a, 1))

    for j in range(num_iterations):
        idx = j % len(examples)

        single_example = examples[idx]
        single_example_char = [c for c in single_example]
        single_example_idx = [char_to_ix[c] for c in single_example]
        X = [None] + single_example_idx

        ix_newline = char_to_ix['\n']
        Y = single_example_idx + [ix_newline]

        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters, learning_rate=0.01)
