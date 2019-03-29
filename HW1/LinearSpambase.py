import math
from random import randrange

import numpy as np
from numpy.linalg import inv
from sklearn import metrics
import matplotlib.pyplot as plt

def main():

    data= '/Users/viveknair/Desktop/ml/hw1/spambase.txt'
    listdata = get_data(data)
    matches_test = []
    matches_train=[]
    # number of features including label
    colcount = len(listdata[0])

    # Convert feature data to float
    for i in range(colcount):
        convert_to_float(listdata, i)
    threshold =0.41
    k_folds = 5
    folds = get_folds(listdata, k_folds)
    train_error = []
    test_error = []
    # for fold in folds:
    fold=folds[3]
    train_data = list(folds)
    train_data.remove(fold)
    test_data=list(fold)
    train_data = sum(train_data, [])

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

    # Calculating w = (X'X)-1 X'Y
    x = np.array(traindata)
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

    mse_train, mse_test = 0.0, 0.0
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
    # Accuracy Measure
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
    # Calc accuracy
    modified_test = assign_labels(predicted_test, threshold)
    match=0
    for i in range(len(actual_test)):
        # print(actual_test[i][0], modified_test[i])
        if int(actual_test[i][0]) == int(modified_test[i]):
            match += 1
    accuracy = match / len(actual_test)
    matches_test.append(accuracy)

    confusion_matrix= create_confusion_mtrx(actual_test, modified_test)
    print(" Confusion Matrix: ", confusion_matrix)

    # print(predicted_test_y)
    pred_list =[]
    for i in range(len(predicted_test)):
        pred_list.append(predicted_test[i][0])
    pred_test = np.array(pred_list)

    fpr, tpr, thresholds = metrics.roc_curve(y2, pred_test)
    print("thresholds: ", thresholds)
    auc = metrics.auc(fpr, tpr)
    plot_roc(fpr, tpr, auc)



    print("Average training error : ", error_avg(train_error))
    print("Average testing error : ", error_avg(test_error))
    print('\n')

    print("Average Accuracy in training: ", error_avg(matches_train))
    print("Average Accuracy in testing: ", error_avg(matches_test))



def error_avg(error):
    sum=0.0
    for err in error:
        sum += err
    return sum/len(error)

def plot_roc(fpr, tpr, auc):
    plt.title('ROC for Linear Regression')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def get_rates(matrix):
    tp = matrix[0][0]
    fn = matrix[1][0]
    fp = matrix[0][1]
    tn =  matrix[1][1]

    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)

    return tpr, fpr
def assign_labels(predicted,threshold):
    mod_labels=[]
    for i in predicted:
        if float(i[0]) >= threshold:
            mod_labels.append(1.0)
        else:
            mod_labels.append(0.0)
    return mod_labels
def get_data(filename):

    listdata = []
    with open(filename, "r") as file:
        data = file.readlines()
        for line in data:
            newline = line.strip().replace("   ", " ").replace("  ", " ").split(',')
            if newline[0] != '':
                listdata.append(newline)

    return listdata


def get_data_without_labels(data):
    for dat in data:
        del dat[-1]
    return data

def add_bias(data):
    for i in range(len(data)):
        data[i] = [1] + data[i]
    return data

def create_confusion_mtrx(actual, predicted):
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


def get_labels(data,colcount):
    labels=[]
    for datapoint in data:
        labels.append(datapoint[colcount-1])
    return labels


def convert_to_float(data,feature):
    for datapoint in data:
        datapoint[feature] = float(datapoint[feature])


def get_normalized_data(minim, origdata, colcount):
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

    return origdata


def get_folds(data, k):
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