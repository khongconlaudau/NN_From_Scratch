import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

train_dataset = h5py.File('datasets/train_signs.h5', 'r')
test_dataset = h5py.File('datasets/test_signs.h5', 'r')

x_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_x'])
y_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_y'])

x_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_x'])
y_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_y'])

# if __name__ == "__main__":
#
#
#     print(x_train.element_spec)
#     print(next(iter(x_train)))
#
#     unique_labels = set()
#     for element in y_train:
#         unique_labels.add(
#             element.numpy())  # element.numpy() converts each element from a TensorFlow tensor to a NumPy type (or a standard Python type).
#     print(unique_labels)
#
#     # Visualize
#     images_iter = iter(x_train)
#     labels_iter = iter(y_train)
#     plt.figure(figsize=(10, 10))
#     for i in range(25):
#         ax = plt.subplot(5, 5, i+1) # is creating a subplot in a 5x5 grid (meaning 5 rows and 5 columns) and selecting the (i + 1)th subplot as the active one.
#         plt.imshow(next(images_iter).numpy().astype("uint8"))
#         plt.title(next(labels_iter).numpy().astype("uint8"))
#         plt.axis('off')
#
#     plt.show()

def normalize(image):
    """
    Transform an image into a tensor of shape (64 * 64 * 3, )
    and normalize its components.

    Arguments
    image - Tensor.

    Returns:
    result -- Transformed tensor
    """
    image = tf.cast(image, tf.float32) / 255.0 #The tf.cast() function in TensorFlow is used to change the data typeof a tensor.
    image = tf.reshape(image, [-1,])
    return image

"""
There's one more additional difference between TensorFlow datasets and Numpy arrays: If you need to transform one, you would invoke the map method to apply the function passed as an argument to each of the elements.
"""

norm_train = x_train.map(normalize)
new_test = x_test.map(normalize)

print(norm_train.element_spec)

def linear_fucntion():
    """
    Implements a linear function:
            Initializes X to be a random tensor of shape (3,1)
            Initializes W to be a random tensor of shape (4,3)
            Initializes b to be a random tensor of shape (4,1)
    Returns:
    result -- Y = WX + b
    """
    np.random.seed(1)

    """
    Note that the difference between tf.constant and tf.Variable is that you can modify the state of a tf.Variable but cannot change the state of a tf.constant.
    """

    X = tf.constant(np.random.randn(3,1), name='X')
    W = tf.Variable(np.random.randn(4,3), name='W')
    b = tf.Variable(np.random.randn(4,1), name='b')

    Y = tf.matmul(X, W) + b
    return Y

def sigmoid(z):
    """
    Computes the sigmoid of z

    Arguments:
    z -- input value, scalar or vector

    Returns:
    a -- (tf.float32) the sigmoid of z


    """
    # tf.keras.activations.sigmoid requires float16, float32, float64, complex64, or complex128.
    z = tf.cast(z, tf.float32)
    a = tf.keras.activations.sigmoid(z)

    return a


"""
Now you'll initialize a vector of numbers with the Glorot initializer. The function you'll be calling is tf.keras.initializers.GlorotNormal, which draws samples from a truncated normal distribution centered on 0, with stddev = sqrt(2 / (fan_in + fan_out)), where fan_in is the number of input units and fan_out is the number of output units, both in the weight tensor.

To initialize with zeros or ones you could use tf.zeros() or tf.ones() instead.
"""
def initialize_parameters():
    """
    Initializes parameters to build a neural network with TensorFlow. The shapes are:
                        W1 : [25, 12288]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [6, 12]
                        b3 : [6, 1]

    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
    initializer = tf.keras.initializers.GlorotNormal(seed=1)
    W1 = tf.Variable(initializer([25, 12288]), name='W1')
    b1 = tf.Variable(initializer([25, 1]), name='b1')
    W2 = tf.Variable(initializer([12, 25]), name='W2')
    b2 = tf.Variable(initializer([12, 1]), name='b2')
    W3 = tf.Variable(initializer([6, 12]), name='W3')
    b3 = tf.Variable(initializer([6, 1]), name='b3')

    parameters = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2,
        'W3': W3,
        'b3': b3,
    }

    return parameters

def foward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.linalg.matmul(W1, X), b1)
    A1 = tf.keras.activations.relu(Z1)
    Z2 = tf.add(tf.linalg.matmul(W2, A1), b2)
    A2 = tf.keras.activations.relu(Z2)
    Z3 = tf.add(tf.linalg.matmul(W3, A2), b3)

    return Z3

def compute_total_lost(logits, labels):
    """
    Computes the total loss

    Arguments:
    logits -- output of forward propagation (output of the last LINEAR unit), of shape (6, num_examples)
    labels -- "true" labels vector, same shape as Z3

    Returns:
    total_loss - Tensor of the total loss value
    """
    total_loss = tf.reduce_sum(tf.keras.losses.categorical_crossentropy(tf.transpose(labels), tf.transpose(logits), from_logits=True))

    return total_loss

def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001, num_epochs= 1500, minibatch_size = 32):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 10 epochs

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    cost, train_acc, test_acc = [], [], []

    parameters = initialize_parameters()

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # The CategoricalAccuracy will track the accuracy for this multiclass problem
    train_accuracy = tf.keras.metrics.CategoricalAccuracy()
    test_accuracy = tf.keras.metrics.CategoricalAccuracy()

    # In your code, the mini-batch creation and iteration through the batches are handled by TensorFlow's tf.data.Dataset
    dataset = tf.data.Dataset.zip((X_train, Y_train))
    test_dataset = tf.data.Dataset.zip((X_test, Y_test))

    # We can get the number of elements of a dataset using the cardinality method
    m = dataset.cardinality().numpy()

    """The prefetch() method in TensorFlow's tf.data API is used to optimize the input pipeline by preparing the next batch of data while the model is training on the current batch. This helps to avoid delays in training due to slow data loading from disk.
    """

    minibatches = dataset.batch(minibatch_size).prefetch(tf.data.experimental.AUTOTUNE)
    test_minibatches = test_dataset.batch(minibatch_size).prefetch(tf.data.experimental.AUTOTUNE)

    for epoch in range(num_epochs):
        epoch_total_loss = 0

        train_accuracy.reset_states()

        for (minibatch_X, minibatch_Y) in minibatches:
            with tf.GradientTape() as tape:
                """
                We need to use tf.GradientTape to track the operations involved in the forward pass (prediction) and loss computation so that we can later compute the gradients for backpropagation. This enables TensorFlow to perform automatic differentiation, which is the key to training neural networks by adjusting the parameters based on the loss
                """
                # predict
                Z3 = foward_propagation(tf.transpose(minibatch_X), parameters)

                # compute loss
                minibatch_total_loss = compute_total_lost(Z3, tf.transpose(minibatch_Y))
            train_accuracy.update_state(minibatch_Y, tf.transpose(Z3))

            trainable_variables = [W1, b1, W2, b2, W3, b3]
            grads = tape.gradient(minibatch_total_loss, trainable_variables)
            optimizer.apply_gradients(zip(grads, trainable_variables))
            epoch_total_loss += minibatch_total_loss

        epoch_total_loss /= m
    return parameters, train_accuracy, test_accuracy

