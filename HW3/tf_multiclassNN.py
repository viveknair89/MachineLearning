from decimal import Decimal

import numpy as np
import tensorflow as tf

"""
    This program implements a multi-class supervised Neural Network with a square loss and a sigmoid activation function
    using tensorFlow library
"""

class multinn():
    """
    This is a multiclass neural network
    """
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

        self.hidden = tf.nn.sigmoid(tf.add(tf.matmul(self.X, self.hidden_w), self.hidden_bias), name='hidden_layer')
        self.output = tf.nn.sigmoid(tf.add(tf.matmul(self.hidden, self.output_w), self.output_bias), name='output_layer')

        self.loss = tf.reduce_mean(tf.pow(self.output - self.Y, 2))

        self.optimizer = tf.train.AdamOptimizer(self.learnrate, name="Gradient_descent").minimize(self.loss)

    def update(self, x, y):
        """
        Optimizes weights each iteration
        :param x: a datapoint
        :param y: a data label
        :return: returns loss and output result values after optimization
        """
        sess = tf.get_default_session()
        _, l, out = sess.run([self.optimizer, self.loss, self.output], feed_dict={self.X: [x], self.Y: [y]})
        return l, out

    def get_prediction(self, x, y):
        """
        predicts output value of test dataset and calculates accuracy
        :param x: test dataset
        :param y: test label vector
        :return: accuracy of prediction
        """
        preds = []
        sess = tf.get_default_session()
        for i,j in zip(x, y):
            p= sess.run(self.output, {self.X: [i]})
            preds.append(get_accuracy(p, j))
        return np.mean(preds)

def main(_):

    # Fetch preprocessed Data
    train_dat, train_labels = read_data("train_wine.csv")
    test_dat, test_labels = read_data("test_wine.csv")
    train_x, test_x = get_normalized_data(train_dat+test_dat, len(train_dat[0]), len(train_dat))

    model = multinn()

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for ind in range(100):
            acc=[]
            # Train the model
            for x, y in zip(train_x, train_labels):
                l, out = model.update(x, y)
                accuracy = get_accuracy(out, y)
                acc.append(accuracy)
                #print(out, y)

            print("Iteration: ", ind)
            print("Training Accuracy: ", np.mean(acc))

            # Predict
            pred = model.get_prediction(test_x, test_labels)
            print("Testing Accuracy: ", pred)
            print()
            print(l)


def read_data(filename):
    """
    read data from file and preprocess it
    :param filename: name of the input file
    :return: processed data and labels
    """
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
    """
    Computes labels based on predicted values
    :param prediction: predicted value
    :param labels: actual labels
    :return: predicted label for the dataset
    """
    if np.argmax(prediction[0]) == np.argmax(labels):
        return 1
    else:
        return 0


def get_normalized_data(origdata, colcount, traincount):
    """
    Normalize the data
    :param traincount: number of records in training dataset
    :param origdata: input train data
    :param colcount: number of features/columns
    :return: normalized data(train and test data)
    """
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
