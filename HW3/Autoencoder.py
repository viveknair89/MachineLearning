import numpy as np

"""
    This program implements an auto-encoder for an 8X8 input vector
"""

def sigmoid(x):
    """
    Computes sigmoid of a given vector
    :param x: input data vector
    :return: sigmoid of input vector
    """
    return 1/(1+np.exp(-x))


def encoder(x):
    """
    Encodes input vector into hidden layer
    :param x: input data vector
    :return: Encoded hidden layer
    """
    hidden_z = np.dot(x, encoder_hidden_wt) + encoder_bias
    layer = sigmoid(hidden_z)
    return layer


def decoder(x):
    """
    Decoded input of given hidden layer
    :param x: encoded hidden layer
    :return: decoded input values
    """
    out_z = np.dot(x, decoder_hidden_wt) + decoder_bias
    layer = sigmoid(out_z)
    return layer

lr = 0.1
num_steps = 350
features= 8
hidden_units= 3

# Defining Input
input = [[1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,], [0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1]]

# Converting into Float
fl_input=[]
for i in input:
    j=[]
    for w in i:
        j.append(float(w))
    fl_input.append(j)

# Defining seed for constant results
np.random.seed(30)

# Initializing hidden weights for encoder and decoder layer
encoder_hidden_wt = np.random.rand(features, hidden_units)
decoder_hidden_wt = np.random.rand(hidden_units, features)

# Initializing bias values for encoder and decoder layer
encoder_bias = np.random.rand(hidden_units)
decoder_bias = np.random.rand(features)



loss= []
acc =[]

# Encoding and Decoding
x_data = np.array(fl_input)
encoded = encoder(np.array(x_data))
decoded = decoder(encoded)

for i in range(num_steps):

    # Feed Forward
    encoded = encoder(np.array(x_data))
    decoded = decoder(encoded)
    hidden_z = np.dot(x_data, encoder_hidden_wt) + encoder_bias
    out_z = np.dot(decoder_hidden_wt, x_data) + decoder_bias

    # Back Propagation
    diff_cost_ow = np.dot(encoded.T, (decoded - x_data))

    diff_cost__hr = np.dot((decoded - x_data), out_z.T)

    diff_rh_zh = sigmoid(hidden_z)*(1-sigmoid(hidden_z))

    diff_cost_hw = np.dot(x_data.T, diff_rh_zh*diff_cost__hr)
    diff_cost_bh = diff_cost__hr*diff_rh_zh

    encoder_hidden_wt = encoder_hidden_wt - lr * diff_cost_hw
    encoder_bias = encoder_bias - lr * diff_cost_bh.sum(axis=0)

    decoder_hidden_wt = decoder_hidden_wt - lr* diff_cost_ow
    decoder_bias = decoder_bias - lr* (decoded - x_data).sum(axis=0)

    # Loss
    l =  ((1 / 2) * (np.power((decoded - x_data), 2)))
    # print("loss :", l.sum())
    loss.append(l)


    # Calculating Accuracy
    acc = 0
    for ind in range(len(decoded)):
        # print(input[ind], decoded[ind])
        if np.argmax(input[ind]) == np.argmax(decoded[ind]):
            acc += 1
    print("Iteration: ", i)
    print("Number of Matching codes: ", acc)
    print()




