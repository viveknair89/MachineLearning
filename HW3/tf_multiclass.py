from decimal import Decimal

import numpy as np

"""
    This program implements a multi-class supervised Neural Network with a square loss and a sigmoid activation function
"""
def main():
    learnrate = 0.1
    hidden_units = 4
    classes =3
    epochs = 100

    # Fetch preprocessed Data
    train_dat, train_labels = read_data("train_wine.csv")
    test_dat, test_labels = read_data("test_wine.csv")

    train_x, test_x = get_normalized_data(train_dat+test_dat, len(train_dat[0]), len(train_dat))
    size = len(train_x)
    features = len(train_x[0])

    # set seed for consistency in results
    np.random.seed(100)

    # Initialize output/hidden layer weights and bias
    hidden_w = np.random.rand(features, hidden_units)
    hidden_bias = np.random.rand(hidden_units)
    output_w = np.random.rand(hidden_units, classes)
    output_bias = np.random.rand(classes)

    loss= []
    acc =[]
    for i in range(epochs):

        # Feed Forward Step
        hidden_z, hidden_res, output_z, output_res = feed_fwd(train_x, hidden_w, hidden_bias, output_w, output_bias)

        # Back Propagation
        # Calculating gradients of output layer
        diff_mse = output_res - train_labels
        diff_ow_oz = sigmoid_deriv(output_z)
        diff_cost_oz = diff_mse * diff_ow_oz
        diff_cost_ow = np.dot(hidden_res.T, diff_cost_oz)

        # Calculating gradients of hidden layer
        diff_cost_hr = np.dot(diff_cost_oz, output_w.T)
        diff_hr_hz = sigmoid_deriv(hidden_z)
        diff_cost_hw = np.dot(train_x.T, diff_hr_hz * diff_cost_hr)
        diff_cost_hb = diff_cost_hr * diff_hr_hz
        # Updating weights
        hidden_w = hidden_w - learnrate * diff_cost_hw
        output_w = output_w - learnrate * diff_cost_ow

        # Updating bias
        hidden_bias = hidden_bias - learnrate * diff_cost_hb.sum(axis=0)
        output_bias = output_bias - learnrate * diff_ow_oz.sum(axis=0)

        # Get Accuracy
        accuracy = get_accuracy(output_res, train_labels)
        acc.append(accuracy)
        if i %10 ==0:
            l =  ((1 / 2) * (np.power((output_res - train_labels), 2)))
            # print("iter: ", i)
            # print("loss  :", l)
            loss.append(l)
        print("Iteration: ", i)
        print("Training Accuracy: ", np.mean(acc))
        print("Testing Accuracy: ", validate(test_x, test_labels, hidden_w, hidden_bias, output_w, output_bias))
        print()


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


def sigmoid_deriv(x):
    """
    Calculates derivative of the sigmoid function for given parameter
    :param x: parameter data
    :return: sigmoid derivative of the input parameter
    """
    return sigmoid(x)* (1-sigmoid(x))


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
    output_res = sigmoid(output_z)
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
    _,_,_,pred = feed_fwd(x, hidden_w, hidden_bias, output_w, output_bias)
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
    minim = []
    # Get minimum value for each feature for normalization
    for col in range(colcount - 1):
        ind = 0
        for datapoint in origdata:
            ind += 1
            if ind == 1:
                minim.append(datapoint[col])
            else:
                if datapoint[col] < minim[col]:
                    minim[col] = datapoint[col]
    maxim = []
    datcnt = 0
    for dataset in origdata:
        datcnt += 1
        for ind in range(colcount - 1):
            dataset[ind] = dataset[ind] - minim[ind]
            if datcnt == 1:
                maxim.append(dataset[ind])
            else:
                if dataset[ind] > maxim[ind]:
                    maxim[ind] = dataset[ind]

    for datapoint in origdata:
        for ind in range(colcount - 1):
            datapoint[ind] = datapoint[ind] / maxim[ind]

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


if __name__ == '__main__':
    main()
