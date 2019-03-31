import math
from random import randrange

import numpy as np
from numpy.linalg import inv

"""
    This program implements Ridge Regression algorithm on Spambase dataset
"""
def main():

    alpha =3
    k_folds = 5
    train_error = []
    test_error = []
    matches_test = []
    matches_train = []
    threshold = 0.45

    # Fetch preprocessed Data
    data= '/Users/viveknair/Desktop/ml/hw1/spambase.txt'
    listdata = get_data(data)

    # number of features including label
    colcount = len(listdata[0])

    # Convert feature data to float
    for i in range(colcount):
        convert_to_float(listdata, i)

    # Get k folded data
    folds = get_folds(listdata, k_folds)

    for fold in folds:
        train_data = list(folds)
        train_data.remove(fold)
        test_data=list(fold)
        train_data = sum(train_data, [])
        print(len(train_data[0]))
        print(len(test_data[0]))

        # Get label vectors for training and test data
        labels = get_labels(train_data, colcount)
        labels_test = get_labels(test_data, colcount)

        train=[]
        test=[]
        # Remove labels from train, test data
        for dat in train_data:
            train.append(dat[:-1])
        for t in test_data:
            test.append(t[:-1])
        print(len(test[1]))
        train, test = get_normalized_data(train, test, colcount)

        # Calculating w = (X'X)-1 X'Y
        x = np.array(train)
        x_test = np.array(test)
        x_transpose = x.transpose()

        x_transpose_x = np.dot(x_transpose, x)

        identity = np.identity(colcount-1)
        alpha_identity = np.dot(alpha, identity)
        x_transpose_x_alpha_id = np.add(x_transpose_x, alpha_identity)

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

        train_error.append(mse_train)
        print("training_error: ", mse_train)
        print("\n")
        # Training Accuracy Measure
        match = 0
        modified_train = assign_labels(predicted, threshold)
        for i in range(len(actual)):
            if int(actual[i][0]) == int(modified_train[i]):
                match += 1
        accuracy = match / len(actual)
        matches_train.append(accuracy)

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

        test_error.append(mse_test)
        print("test error: ", mse_test)
        print("\n")
        # Calc Testing accuracy
        modified_test = assign_labels(predicted_test, threshold)
        match = 0
        for i in range(len(actual_test)):
            if int(actual_test[i][0]) == int(modified_test[i]):
                match += 1
        accuracy = match / len(actual_test)
        matches_test.append(accuracy)
        print("\n")


    print("Average training error : ", error_avg(train_error))
    print("Average testing error : ", error_avg(test_error))

    print("Average Accuracy in training: ", error_avg(matches_train))
    print("Average Accuracy in testing: ", error_avg(matches_test))




def error_avg(error):
    """
    Calculates Average Error
    :param error: vector of errors
    :return: Average Error
    """
    sum=0.0
    for err in error:
        sum += err
    return sum/len(error)


def get_data(filename):
    """
    Get Preprocessed data
    :param filename: file name of input data
    :return: processed data
    """
    listdata = []
    with open(filename, "r") as file:
        data = file.readlines()
        for line in data:
            newline = line.strip().replace("   ", " ").replace("  ", " ").split(',')
            if newline[0] != '':
                listdata.append(newline)

    return listdata


def assign_labels(predicted,threshold):
    """
    Assign labels according to threshold
    :param predicted: predicted values vector
    :param threshold: threshold value for deciding labels
    :return: predicted labels vector
    """
    mod_labels=[]
    for i in predicted:
        if float(i[0]) >= threshold:
            mod_labels.append(1.0)
        else:
            mod_labels.append(0.0)
    return mod_labels


def get_data_without_labels(data):
    """
    Get labels removed from the input Data
    :param data: input data
    :return: data without labels
    """
    for dat in data:
        del dat[-1]
    return data

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


def get_folds(data, k):
    """
    Split data into k folds
     :param data: whole input data set
     :param k: number of folds to be split into
     :return: data divided randomly into k folds
     """
    split_data = []
    fold_size = int(len(data) / k)
    for i in range(k):
        fold = []
        while len(fold) < fold_size:
            ind = randrange(len(data))
            fold.append(data.pop(ind))
        split_data.append(fold)
    return split_data


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