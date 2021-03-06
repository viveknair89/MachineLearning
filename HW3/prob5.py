from decimal import Decimal

import numpy as np

"""
    This program implements a multi-class supervised Neural Network using likelihod objective with cross entropy for loss
    function, softmax activation for output layer and sigmoid activation for hidden layer
"""

def main():

    learnrate = 0.1
    hidden_units = 13
    classes =3
    epochs = 400

    # Fetch preprocessed Data
    train_dat, train_labels = read_data("train_wine.csv")
    test_dat, test_labels = read_data("test_wine.csv")
    train_x, test_x = get_normalized_data(train_dat+test_dat, len(train_dat[0]), len(train_dat))
    size = len(train_x)
    features = len(train_x[0])
    # print(train_labels.shape)

    # Set seed for consistency in results
    np.random.seed(3)

    # Initializing hidden and output layer weights and bias
    hidden_w = np.float64(np.random.rand(features, hidden_units))
    hidden_bias = np.float64(np.random.rand(hidden_units))

    output_w = np.float64(np.random.rand(hidden_units, classes))
    output_bias = np.float64(np.random.rand(classes))

    loss= []
    acc =[]
    for i in range(epochs):

        # Feed Forward
        hidden_z, hidden_res, output_z, output_res = feed_fwd(train_x, hidden_w, hidden_bias, output_w, output_bias)

        # Back Propagation
        # cross_entropy = (output_res - train_labels)/len(train_x)
        cross_entropy = (output_res - train_labels)
        diff_cost_ow = np.dot(hidden_res.T, cross_entropy)

        diff_cost__hr = np.dot(cross_entropy, output_w.T)

        diff_rh_zh = sigmoid_deriv(hidden_z)

        diff_cost_hw = np.dot(train_x.T, diff_rh_zh*diff_cost__hr)
        diff_cost_bh = diff_cost__hr*diff_rh_zh

        hidden_w = hidden_w - learnrate * diff_cost_hw
        hidden_bias = hidden_bias - learnrate * diff_cost_bh.sum(axis=0)

        output_w = output_w - learnrate * diff_cost_ow
        output_bias = output_bias - learnrate * cross_entropy.sum(axis=0)

        accuracy = get_accuracy(output_res, train_labels)
        acc.append(accuracy)
        # if i %10 ==0:
        l =  np.sum(-train_labels * np.log(output_res))
        # print("iter: ", i)
        # print("loss  :", l.sum())
        # loss.append(l)
        print("Iteration: ", i)
        print("Training Accuracy: ", np.mean(acc))
        print("Testing Accuracy: ", validate(test_x, test_labels, hidden_w, hidden_bias, output_w, output_bias))
        print()


def get_accuracy(prediction, labels):
    """
    Calculates accuracy of predicted labels
    :param prediction: predicted values vector
    :param labels: actual labels
    :return: predicted label value
    """
    # acc=0
    # for x,y in zip(prediction, labels):
    #     if np.argmax(prediction) == np.argmax(labels):
    #         acc+=1
    # return acc/len(labels)
    if np.argmax(prediction[0]) == np.argmax(labels):
        return 1
    else:
        return 0


def sigmoid_deriv(x):
    """
    Calculates derivative of the sigmoid function for given parameter
    :param x: parameter data
    :return: sigmoid derivative of the input parameter
    """
    np.multiply.reduce(np.arange(21) + 1)
    return sigmoid(x) * (1-sigmoid(x))


def feed_fwd(train_x, hidden_w, hidden_bias, output_w, output_bias):
    """
    Executes feed forward step in the algorithm for building neural network
    :param train_x: train data
    :param hidden_w: hidden layer weights
    :param hidden_bias: hidden layer bias
    :param output_w: output layer weights
    :param output_bias: output layer bias
    :return: updated hidden, output layer values
    """
    hidden_z = np.dot(train_x, hidden_w) + hidden_bias

    hidden_res = sigmoid(hidden_z)

    output_z = np.dot(hidden_res, output_w) + output_bias
    output_res = softmax(output_z)

    return hidden_z, hidden_res, output_z, output_res


def validate(x, y, hidden_w, hidden_bias, output_w, output_bias):
    """
    models the test dataset in the built neural network and predicts label values
    :param x: test dataset
    :param y: test labels
    :param hidden_w: hidden layer weights of neural network
    :param hidden_bias: hidden layer bias values of neural network
    :param output_w: output layer weights of neural network
    :param output_bias: output layer bias of neural network
    :return: predicted values vector
    """
    preds=[]
    # for i, j in zip(x,y):
    hidden_z, hidden_res, output_z, pred = feed_fwd(x, hidden_w, hidden_bias, output_w, output_bias)
    preds.append(get_accuracy(pred, y))
    return np.mean(preds)

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
            data.append([float(i) for i in newline[1:]])
            labels.append(m[newline[0]])
    return data, np.array(labels)

def get_normalized_data(origdata, colcount, traincount):
    """
    Normalize the data
    :param traincount: number of records in training dataset
    :param origdata: input train data
    :param colcount: number of features/columns
    :return: normalized data(train and test data)
    """
    minim = list(map(min, *origdata))
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


def sigmoid(x):
    """
    Calculates sigmoid of given parameter
    :param x: input data parameter
    :return: sigmoid fun of param x
    """
    for i in x:
        for j in i:
            j = Decimal(1.0)/(Decimal(1.0) + Decimal(np.exp(-j)))
            # j[j == 1.0] = 0.9999
            # j[j == 0.0] = 0.0001
    return x
    # return 1/(1+np.exp(-x))

def softmax(x):
    """
    Calculates softmax value of given parameter
    :param x: input data parameter
    :return: softmax fun of param x
    """
    # expx = np.exp(x - np.max(x, axis=1, keepdims=True))
    # return expx / expx.sum(expx, axis=1, keepdims=True)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

if __name__ == '__main__':
    main()
