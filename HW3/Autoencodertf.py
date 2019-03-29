import tensorflow as tf
import numpy as np

input = [[1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,], [0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1]]
fl_input=[]
for i in input:
    j=[]
    for w in i:
        j.append(float(w))
    fl_input.append(j)
lr = 0.01
num_steps = 500
features= 8
hidden_units= 3

X = tf.placeholder('float32', [None, features], name='features')

encoder_hidden_wt = tf.Variable(tf.random_normal([features, hidden_units]))
print("encoder:", encoder_hidden_wt)
decoder_hidden_wt = tf.Variable(tf.random_normal([hidden_units, features]))

encoder_bias = tf.Variable(tf.random_normal([hidden_units]))
decoder_bias = tf.Variable(tf.random_normal([features]))


def encoder(x):
    x=tf.cast(x, tf.float32)
    # print("encoder shape:",x.shape)
    layer = tf.nn.sigmoid(tf.add(tf.matmul(x, encoder_hidden_wt), encoder_bias))
    return layer

def decoder(x):
    layer = tf.nn.sigmoid(tf.add(tf.matmul(x, decoder_hidden_wt), decoder_bias))
    return layer
print(fl_input)
X_data= np.array(fl_input)
encoded = encoder(np.array(X_data))
decoded = decoder(encoded)

loss = tf.reduce_mean(tf.pow(X_data-decoded, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(loss)

init = tf.global_variables_initializer()

# Start a new tf session
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    for ind in range(num_steps):
        acc = []
        _, l = sess.run([optimizer, loss], feed_dict={X: X_data})
        print("Iteration: ", ind)
        print("training loss: ", l)
    g = sess.run(decoded, feed_dict={X: X_data})
    acc =0
    for ind in range(len(g)):
        # print(input[ind], g[ind])
        if np.argmax(input[ind]) == np.argmax(g[ind]):
            acc+=1

    print("Number of Matching codes: ", acc)
    print()
