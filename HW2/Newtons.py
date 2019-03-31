import math
from random import randrange

import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

"""
    This program builds a Logistic regression training model using Newton's Numerical training method
    on the Spambase dataset
"""
def main():
    epoch = 100
    k_folds = 5
    train_error = []
    test_error = []

    # Fetch data and do preprocessing
    data = '/Users/viveknair/Desktop/ml/hw1/spambase.txt'
    listdata = get_data(data)

    # number of features including label
    colcount = len(listdata[0])

    # Convert feature data to float
    for i in range(colcount):
        convert_to_float(listdata, i)


    folds = get_folds(listdata, k_folds)

    for fold in folds:
    # fold = folds[0]
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
        w = np.zeros(x.shape[1])

        #  wNew = wOld - H^-1 * <partial derivative of f with respect to w(Grad(J)>
        for i in range(epoch):
            z = np.dot(x, w)
            gz = 1 / (1 + np.exp(-z))
            p = gz.tolist()
            gradJ = []
            for feature in range(colcount):
                val =0.0
                for datapnt in range(len(x)):
                    val += x[datapnt][feature] * (float(labels[datapnt])-p[datapnt])
                gradJ.append(-val)
            gradient = np.array(gradJ)
            print(gradient.shape)
            hess= []
            for ind in range(colcount):
                hess.append(build_hess(ind, colcount, p, x))
            hessian = np.array(hess)
            print(hessian.shape)
            w = w - np.linalg.pinv(hessian).dot(gradient)


        # Training error calculation
        preds = get_predicts(x, w)
        predicted = preds.tolist()
        actual = y.tolist()
        print("predicted training label", predicted)
        print("actual training label", labels)
        acc = 0.0
        for i in range(len(actual)):
            acc += math.pow((predicted[i] - float(labels[i])), 2)
        mse_train = acc / len(actual)

        train_error.append(mse_train)
        print("training_error: ", mse_train)
        print("\n")

        #   Testing error calculation
        pred = get_predicts(x_test, w)
        predicted_test = pred.tolist()
        actual_test = y_test.tolist()
        print("predicted test label", predicted_test)
        print("actual test label", labels_test)
        acc = 0.0
        for i in range(len(actual_test)):
            acc += math.pow((predicted_test[i] - float(labels_test[i])), 2)
        mse_test = acc / len(actual_test)
        test_error.append(mse_test)
        print("test_error: ", mse_test)
        print("\n")

        # Create confusion Matrix
        confusion_matrix = create_confusion_mtrx(actual_test, predicted_test)
        print(confusion_matrix)

        pred_test = np.array(predicted_test)
        print(y2)
        print(pred_test)
        fpr, tpr, thresholds = metrics.roc_curve(y2, pred_test)
        print("fpr: ", fpr)
        print("fpr: ", tpr)
        print("thresholds: ", thresholds)

        # Calculate AUC and plot ROC
        auc = metrics.auc(fpr, tpr)
        plot_roc(fpr, tpr, auc)

def error_avg(error):
    """
    Calculates Average Error
    :param error: vector of errors
    :return: Average Error
    """
    sum = 0.0
    for err in error:
        sum += err
    return sum / len(error)

def build_hess(col,colcount, p, x):
    """
    Build Hessian Matrix
    :param col: feature data
    :param colcount: number of features
    :param p: Gradient
    :param x: input data
    :return: Hessian matrix
    """
    row=[]
    for i in range(colcount):
        acc =0.0
        for data in range(len(x)):
            acc += x[data][i]*p[data]*(1-p[data])*x[data][col]
        row.append(acc)
    return row

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


def get_labels(data, colcount):
    """
    Returns labels (last column) from the given data
    :param data: input dataset
    :param colcount: column count
    :return: list of labels extracted from input data
    """
    labels = []
    for datapoint in data:
        labels.append(datapoint[colcount - 1])
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


def convert_to_float(data, feature):
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
    :return: normalized data(train, test)
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


def plot_roc(fpr, tpr, auc):
    """
    Plot ROC Curve
    :param fpr: False positive rate
    :param tpr: true positive rate
    :param auc: Area under the curve
    """
    plt.title('ROC for Logistic Regression (Newtons)')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


if __name__ == '__main__':
    main()
