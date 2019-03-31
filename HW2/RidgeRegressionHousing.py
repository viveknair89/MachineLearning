import math
import numpy as np
from numpy.linalg import inv

"""
    This program implements Ridge Regression algorithm on Housing Data
"""

def main():

    # Fetch preprocessed training and test data
    train_data= '/Users/viveknair/Desktop/ml/hw1/housing_train.txt'
    listdata = get_data(train_data)
    test_data = '/Users/viveknair/Desktop/ml/hw1/housing_test.txt'
    testdata = get_data(test_data)
    alpha = [-5, -2, -1, 1, 5, 10, 0.001]
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

    # Get normalized data
    listdata, testdata = get_normalized_data(listdata, testdata, colcount)

    x = np.array(listdata)
    x_test = np.array(testdata)
    x_transpose = x.transpose()

    x_transpose_x = np.dot(x_transpose, x)

    identity = np.identity(colcount-1)

    for alp in alpha:
        print("\n")
        print(" Predictions for alpha = ", alp)
        alpha_identity = np.dot(alp, identity)
        x_transpose_x_alpha_id = np.add(x_transpose_x, alpha_identity)

        # x_transpose_x_inv = inv(x_transpose_x)

        invfn = inv(x_transpose_x_alpha_id)

        y1= np.array(labels)
        y = np.reshape(y1, (len(labels), 1))

        y2= np.array(labels_test)
        y_test = np.reshape(y2, (len(labels_test), 1))

        x_transpose_y = np.dot(x_transpose, y)

        w = np.dot(invfn, x_transpose_y)

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


def get_normalized_data(origdata, test, colcount):
    """
    Normalize the data
    :param test: test data
    :param origdata: input train data
    :param colcount: number of features/columns
    :return: normalized data(train and test data)
    """
    datacount = len(origdata)
    mean=[]
    combineddata = origdata + test
    for ind in range(colcount - 1):
        acc =0.0
        for dataset in combineddata:
            acc += dataset[ind]
        mean.append(acc)

    for data in combineddata:
        for ind in range(colcount-1):
            data[ind] = data[ind] - mean[ind]

    return combineddata[:datacount], combineddata[datacount:]


if __name__ == '__main__':
    main()