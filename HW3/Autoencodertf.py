import tensorflow as tf
import numpy as np

"""
    This program implements an auto encoder using TensorFlow library for an 8X8 input vector
"""


def encoder(x):
    """
    Encodes input data into hidden layer
    :param x: input data
    :return: encoded hidden layer
    """
    x = tf.cast(x, tf.float32)
    layer = tf.nn.sigmoid(tf.add(tf.matmul(x, encoder_hidden_wt), encoder_bias))
    return layer


def decoder(x):
    """
    Decodes hidden layer back to input data
    :param x: encoded data
    :return: decoded input data
    """
    layer = tf.nn.sigmoid(tf.add(tf.matmul(x, decoder_hidden_wt), decoder_bias))
    return layer


lr = 0.01
num_steps = 500
features= 8
hidden_units= 3

# Input Vector
input = [[1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,], [0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1]]
fl_input=[]

# Convert input into float
for i in input:
    j=[]
    for w in i:
        j.append(float(w))
    fl_input.append(j)

# Place Holder for input data
X = tf.placeholder('float32', [None, features], name='features')

# Initializing weights for encoder and decoder for the hidden layer
encoder_hidden_wt = tf.Variable(tf.random_normal([features, hidden_units]))
decoder_hidden_wt = tf.Variable(tf.random_normal([hidden_units, features]))

# Initializing bias variables for encoder and decoder
encoder_bias = tf.Variable(tf.random_normal([hidden_units]))
decoder_bias = tf.Variable(tf.random_normal([features]))

# Encoding and Decoding data
X_data= np.array(fl_input)
encoded = encoder(np.array(X_data))
decoded = decoder(encoded)

# Calculating loss and optimizing model
loss = tf.reduce_mean(tf.pow(X_data-decoded, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(loss)

# Tensorflow operations
init = tf.global_variables_initializer()

# Start a new tf session
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    # Iteratively optimize the losses to obtain perfect encoding/decoding mechanism
    for ind in range(num_steps):
        acc = []
        _, l = sess.run([optimizer, loss], feed_dict={X: X_data})
        print("Iteration: ", ind)
        print("training loss: ", l)

    g = sess.run(decoded, feed_dict={X: X_data})

    # Calculating accuracy of the decoded input
    acc =0
    for ind in range(len(g)):
        if np.argmax(input[ind]) == np.argmax(g[ind]):
            acc+=1

    print("Number of Matching codes: ", acc)
    print()
