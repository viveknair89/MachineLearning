import math

import numpy as np
from numpy.linalg import inv

"""
    This program implements Linear Regression algorithm on Housing Data
"""
def main():

    # Fetch train and test data and preprocess it
    train_data= '/Users/viveknair/Desktop/ml/hw1/housing_train.txt'
    listdata = get_data(train_data)
    test_data = '/Users/viveknair/Desktop/ml/hw1/housing_test.txt'
    testdata = get_data(test_data)

    # number of features including label
    colcount = len(listdata[0])

    # Convert feature data to float
    for i in range(colcount):
        convert_to_float(listdata, i)
        convert_to_float(testdata, i)

    # Get label vectors for training and test data
    labels = get_labels(listdata, colcount)
    labels_test = get_labels(testdata, colcount)

    # Remove labels from train, test data
    for dat in listdata:
        del dat[-1]
    for test in testdata:
        del test[-1]

    # Add bias to data
    listdata = add_bias(listdata)
    testdata = add_bias(testdata)

    # Calculating w = (X'X)-1 X'Y
    x = np.array(listdata)
    x_test = np.array(testdata)
    x_transpose = x.transpose()

    x_transpose_x = np.dot(x_transpose, x)

    x_transpose_x_inv = inv(x_transpose_x)

    y1= np.array(labels)
    y = np.reshape(y1, (len(labels), 1))

    y2= np.array(labels_test)
    y_test = np.reshape(y2, (len(labels_test), 1))

    x_transpose_y = np.dot(x_transpose, y)

    w = np.dot(x_transpose_x_inv, x_transpose_y)

    # Training error calculation
    predicted_y = np.dot(x, w)

    predicted = predicted_y.tolist()
    actual = y.tolist()
    print("predicted training label", predicted)
    print("actual training label", actual)
    acc=0.0
    for i in range(len(actual)):
        acc += math.pow((predicted[i][0]-actual[i][0]), 2)
    mse_train = acc/len(actual)

    print("training_error: ", mse_train)
    print("\n")

    #   Testing error calculation
    predicted_test_y = np.dot(x_test, w)
    predicted_test = predicted_test_y.tolist()
    actual_test = y_test.tolist()
    print("predicted test label", predicted_test)
    print("actual test label", actual_test)
    acc1 = 0.0
    for i in range(len(actual_test)):
        acc1 += math.pow((predicted_test[i][0] - actual_test[i][0]), 2)
    mse_test = acc1 / len(actual_test)

    print("test error: ", mse_test)


def get_data(filename):
    """
    Get Preprocessed data
    :param filename: file name of input data
    :return: processed data
    """
    listdata =[]
    with open(filename, 'r') as file:
        data = file.readlines()
        for line in data:
            newline = line.strip().replace('   ', ' ').replace('  ', ' ').split(' ')
            if newline[0] != '':
                listdata.append(newline)

    return listdata


def add_bias(data):
    """
    Adds bias column to data
    :param data: Input data
    :return: data with bias
    """
    for i in range(len(data)):
        data[i] = [1] + data[i]
    return data


def get_labels(data,colcount):
    """
    Returns labels (last column) from the given data
    :param data: input dataset
    :param colcount: column count
    :return: list of labels extracted from input data
    """
    labels=[]
    for datapoint in data:
        labels.append(datapoint[colcount-1])
    return labels


def convert_to_float(data,feature):
    """
    Converts data to float
    :param data: input data
    :param feature: number of features
    """
    for datapoint in data:
        datapoint[feature] = float(datapoint[feature])


if __name__ == '__main__':
    main()