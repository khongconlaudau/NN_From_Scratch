from sunau import Au_read

import numpy as np

def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image,
    as illustrated in Figure 1.

    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions

    Returns:
    X_pad -- padded image of shape (m, n_H + 2 * pad, n_W + 2 * pad, n_C)
    """

    X_pad = np.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)), mode='constant', constant_values=0)
    return X_pad

def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation
    of the previous layer.

    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)

    Returns:
    Z -- a scalar value, the result of convolving the sliding window (W, b) on a slice x of the input data
    """

    Z = np.sum(a_slice_prev * W) + float(b)
    return Z

def conv_foward(A_prev, W, b, hparams):
    """
       Implements the forward propagation for a convolution function

       Arguments:
       A_prev -- output activations of the previous layer,
           numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
       W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
       b -- Biases, numpy array of shape (1, 1, 1, n_C)
       hparameters -- python dictionary containing "stride" and "pad"

       Returns:
       Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
       cache -- cache of values needed for the conv_backward() function
       """

    # Calculate shape for the current layer
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f,f, n_C_prev, n_C) = W.shape
    stride = hparams['stride']
    pad = hparams['pad']

    n_H = int((n_H_prev + 2 * pad - f) / stride) + 1
    n_W = int((n_W_prev + 2 * pad - f) / stride) + 1

    # Initialize the output volume Z with zeros ( current layer)
    Z = np.zeros((m, n_H, n_W, n_C))

    # apply padding for prev_layer
    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):
            vertical_s = h * stride
            vertical_e = vertical_s + f
            for w in range(n_W):
                horizontal_s = w * stride
                horizontal_e = horizontal_s + f
                for c in range(n_C): # number of filters
                    a_slice_prev = a_prev_pad[vertical_s:vertical_e, horizontal_s:horizontal_e, :]
                    weights = W[:,:,:,c]
                    biases = b[0,0,0,c]
                    Z[i,h,w,c] = conv_single_step(a_slice_prev, weights, biases)

    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparams)
    return Z, cache

def pool_forward(A_prev, hparams, mode = "max"):
    """
    Implements the forward pass of the pooling layer

    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters -- python dictionary containing "f" and "stride"
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")

    Returns:
    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters
    """


    # Calculate the volume of the current layer
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    f = hparams['f']
    stride = hparams['stride']

    n_H = int((n_H_prev - f) / stride) + 1
    n_W = int((n_W_prev - f) / stride) + 1
    n_C = n_C_prev

    # initialize shape of the output
    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_H):
            vert_s = h * stride
            vert_e = vert_s + f
            for w in range(n_W):
                horiz_s = w * stride
                horiz_e = horiz_s + f
                for c in range(n_C):
                    a_slice_prev = a_prev[vert_s:vert_e, horiz_s:horiz_e, c]
                    if mode == "max":
                        A[m,h,w,c] = np.max(a_slice_prev)
                    elif mode == "average":
                        A[m,h,w,c] = np.mean(a_slice_prev)
    cache = (A_prev, hparams)
    return A, cache

def cov_backward(dZ, cache):
    """
    Implement the backward propagation for a convolution function

    Arguments:
    dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward(), output of conv_forward()

    Returns:
    dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
               numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    dW -- gradient of the cost with respect to the weights of the conv layer (W)
          numpy array of shape (f, f, n_C_prev, n_C)
    db -- gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (1, 1, 1, n_C)
    """

    (A_prev, W, b, hparams) = cache
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape

    stride = hparams['stride']
    pad = hparams['pad']

    # Retrieve shape of dZ = Z
    Z, _ = conv_foward(A_prev, W, b, hparams)
    (m, n_H, n_W, n_C) = Z.shape

    # Initialize dA_prev, dW, db with the correct shapes
    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    # Pad A_prev and dA_prev
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_s = h * stride
                    vert_e = vert_s + f
                    horiz_s = w * stride
                    horiz_e = horiz_s + f

                    a_slice = a_prev_pad[vert_s:vert_e, horiz_s:horiz_e,:]

                    da_prev_pad[vert_s:vert_e, horiz_s:horiz_e,:] += W[:,:,:,c] * dZ[i,h,w,c]
                    dW[:,:,:,c] += a_slice * dZ[i,h,w,c]
                    db[:,:,:,c] += dZ[i,h,w,c]

        dA_prev = da_prev_pad[pad:-pad, pad:-pad, :]

    return dA_prev, dW, db

