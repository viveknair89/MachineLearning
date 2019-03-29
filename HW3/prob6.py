from decimal import Decimal

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

class multinn():
    def __init__(self):
        self.learnrate =0.1
        self.hidden_units = 13
        self.inlen = 13
        self.outlen = 3

        self.X = tf.placeholder('float32', [None, self.inlen], name='features')
        self.Y = tf.placeholder('float32', [None, self.outlen], name='labels')

        self.hidden_w = tf.Variable(tf.random_normal([self.inlen, self.hidden_units], seed=tf.set_random_seed(110)),
                            name='hidden_w')
        self.hidden_bias = tf.Variable(tf.random_normal([self.hidden_units], seed=tf.set_random_seed(110)),
                            name='hidden_bias')
        self.output_w = tf.Variable(tf.random_normal([self.hidden_units, self.outlen], seed=tf.set_random_seed(110)),
                            name='output_w')
        self.output_bias = tf.Variable(tf.random_normal([self.outlen], seed=tf.set_random_seed(110)),
                            name='output_bias')

        self.hidden = tf.nn.relu(tf.add(tf.matmul(self.X, self.hidden_w), self.hidden_bias), name='hidden_layer')
        self.output = tf.nn.softmax(tf.add(tf.matmul(self.hidden, self.output_w), self.output_bias), name='output_layer')

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.output))

        self.optimizer = tf.train.AdamOptimizer(self.learnrate, name="AdamOptimizer").minimize(self.loss)

    def update(self, x, y):
        sess = tf.get_default_session()
        # hidden = sess.run([self.hidden_w], feed_dict = {self.X:[x]})
        # bias = sess.run([self.hidden_bias])
        # print(np.add(np.dot([x], hidden), bias))
        # print(sess.run([self.hidden], feed_dict={self.X: [x]}))
        _, l, out = sess.run([self.optimizer, self.loss, self.output], feed_dict={self.X: [x], self.Y: [y]})
        # print(l)
        return l, out

    def get_prediction(self, x, y):
        preds = []
        sess = tf.get_default_session()
        for i,j in zip(x, y):
            p= sess.run(self.output, {self.X: [i]})
            preds.append(get_accuracy(p, j))
        return np.mean(preds)

def main(_):
    train_dat, train_labels = read_data("train_wine.csv")
    test_dat, test_labels = read_data("test_wine.csv")
    train_x, test_x = get_normalized_data(train_dat+test_dat, len(train_dat[0]), len(train_dat))

    model = multinn()

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for ind in range(50):
            acc=[]
            for x, y in zip(train_x, train_labels):
                l, out = model.update(x, y)
                accuracy = get_accuracy(out, y)
                acc.append(accuracy)
                #print(out, y)

            print("Iteration: ", ind)
            print("Training Accuracy: ", np.mean(acc))
            pred = model.get_prediction(test_x, test_labels)
            print("Testing Accuracy: ", pred)
            print()
            # print(l)


def read_data(filename):
    data, labels = [], []
    m = {'1': [1.0, 0.0, 0.0], '2': [0.0, 1.0, 0.0], '3': [0.0, 0.0, 1.0]}
    with open(filename, "r") as file:
        dat = file.readlines()

        for line in dat:
            newline = line.strip().replace("   ", " ").replace("  ", " ").split(',')
            data.append([Decimal(i) for i in newline[1:]])
            labels.append(m[newline[0]])
    # print("length", len(data))
    # print(data)
    return data, np.array(labels)



def get_accuracy(prediction, labels):
    # print(np.argmax(prediction[0]), np.argmax(labels))
    # print(prediction[0])
    # print(labels)
    if np.argmax(prediction[0]) == np.argmax(labels):
        return 1
    else:
        return 0


def get_normalized_data(origdata, colcount, traincount):
    minim =list(map(min, *origdata))
    datcnt = 0
    for dataset in origdata:
        datcnt += 1
        for ind in range(colcount):
            dataset[ind] = dataset[ind] - minim[ind]
    maxim = list(map(max, *origdata))
    minim = list(map(min, *origdata))

    for datapoint in origdata:
        for ind in range(colcount):
            datapoint[ind] = datapoint[ind] / (maxim[ind] - minim[ind])

    return np.array(origdata[:traincount]), np.array(origdata[traincount:])


if __name__ == '__main__':
    tf.app.run()
