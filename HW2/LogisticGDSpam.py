import math
from random import randrange

import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
"""
    This program implements a Logistic regression model using Gradient Descent
"""
def main():
    matches_test = []
    matches_train = []
    threshold = 0.50
    learnrate = 0.001
    epoch = 700
    k_folds = 5
    train_error = []
    test_error = []

    # Fetch Dataset to work on
    data= '/Users/viveknair/Desktop/ml/hw1/spambase.txt'
    listdata = get_data(data)

    # number of features including label
    colcount = len(listdata[0])

    # Convert feature data to float
    for i in range(colcount):
        convert_to_float(listdata, i)

    folds = get_folds(listdata, k_folds)

    # for fold in folds:
    fold=folds[3]
    train_data = list(folds)
    train_data.remove(fold)
    test_data = list(fold)
    train_data = sum(train_data, [])

    train_data, test_data = get_normalized_data(train_data + test_data, colcount, len(train_data))

    # Get label vectors for training and test data
    labels = get_labels(train_data, colcount)
    labels_test = get_labels(test_data, colcount)

    # Add bias to data
    traindata = add_bias(train_data)
    testdata = add_bias(test_data)

    # Remove labels from train, test data
    for dat in traindata:
        del dat[-1]
    for test in testdata:
        del test[-1]

    x = np.array(traindata)
    print(x.shape)
    x_test = np.array(testdata)

    y1 = np.array(labels)
    y = np.reshape(y1, (len(labels), 1))

    y2 = np.array(labels_test)
    y_test = np.reshape(y2, (len(labels_test), 1))
    w = np.zeros(x.T.shape)

    # Build logistic Regression model
    for i in range(epoch):
        z = np.dot(x, w)
        gz = 1/(1+np.exp(-z))
        w = w - learnrate * np.dot(x.T, (gz-y))

    # Training error calculation
    preds = get_predicts(x, w)
    predicted = preds.tolist()
    actual = y.tolist()
    acc=0.0

    # Calculate Training Error
    for i in range(len(actual)):
        acc += math.pow((predicted[i][0]-float(labels[i])), 2)
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
    # print("Training Accuracy: ", accuracy)

    #   Testing error calculation
    preds = get_predicts(x_test, w)
    predicted_test = preds.tolist()
    actual_test = y_test.tolist()

    # Testing Error
    acc = 0.0
    for i in range(len(actual_test)):
        acc += math.pow((predicted_test[i][0] - float(labels_test[i])), 2)
    mse_test = acc / len(actual_test)

    test_error.append(mse_test)
    print("test_error: ", mse_test)
    print("\n")

    # Calc Testing accuracy
    modified_test = assign_labels(predicted_test, threshold)
    match = 0
    for i in range(len(actual_test)):
        if int(actual_test[i][0]) == int(modified_test[i]):
            match += 1
    accuracy = match / len(actual_test)
    # print("Testing Accuracy: ", accuracy)
    matches_test.append(accuracy)

    #  Create Confusion Matrix
    confusion_matrix = create_confusion_mtrx(actual_test, modified_test)
    print(" Confusion Matrix: ", confusion_matrix)

    pred_list = []
    for i in range(len(predicted_test)):
        pred_list.append(predicted_test[i][0])
    pred_test = np.array(pred_list)

    fpr, tpr, thresholds = metrics.roc_curve(y2, pred_test)
    # Calculate AUC and plot ROC curve
    auc = metrics.auc(fpr, tpr)
    plot_roc(fpr, tpr, auc)


    print("Average training error : ", error_avg(train_error))
    print("Average testing error : ", error_avg(test_error))
    print('\n')

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


def plot_roc(fpr, tpr, auc):
    """
    Plot ROC Curve
    :param fpr: False positive rate
    :param tpr: true positive rate
    :param auc: Area under the curve
    """
    plt.title('ROC for Logistic Regression (Normal Eqns)')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

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

def get_predicts(x, w):
    """
    Return the predicted values
    :param x: input data array
    :param w: gradient
    :return:
    """
    z = np.dot(x, w)
    gz = 1 / (1 + np.exp(-z))
    return gz.round()


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


def create_confusion_mtrx(actual, predicted):
    """
    Creates confusion matrix
    :param actual: actual labels
    :param predicted: predicted labels
    :return: Confusion Matrix
    """
    true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0
    conf_matrix=[]
    for i in range(len(actual)):
        if int(actual[i][0]) == int(predicted[i]):
            if int(predicted[i]) == 1:
                true_positive+=1
            else:
                true_negative+=1
        else:
            if int(predicted[i]) == 1:
                false_positive+=1
            else:
                false_negative+=1
    conf_matrix.append([true_positive, false_positive])
    conf_matrix.append([false_negative, true_negative])
    return conf_matrix


def convert_to_float(data,feature):
    """
    Converts data to float
    :param data: input data
    :param feature: number of features
    """
    for datapoint in data:
        datapoint[feature] = float(datapoint[feature])


def get_normalized_data(origdata, colcount, traincount):
    """
    Normalize the data
    :param traincount: number of rows in training data
    :param origdata: input data
    :param colcount: number of features/columns
    :return: normalized data
    """
    minim=[]
    # Get minimum value for each feature for normalization
    for col in range(colcount-1):
        ind=0
        for datapoint in origdata:
            ind += 1
            if ind == 1:
                minim.append(datapoint[col])
            else:
                if datapoint[col] < minim[col]:
                    minim[col] = datapoint[col]
    maxim =[]
    datcnt =0
    for dataset in origdata:
        datcnt += 1
        for ind in range(colcount-1):
            dataset[ind]= dataset[ind]-minim[ind]
            if datcnt == 1:
                maxim.append(dataset[ind])
            else:
                if dataset[ind] > maxim[ind]:
                    maxim[ind] = dataset[ind]

    for datapoint in origdata:
        for ind in range(colcount-1):
            datapoint[ind] = datapoint[ind]/maxim[ind]

    return origdata[:traincount], origdata[traincount:]


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



if __name__ == '__main__':
    main()