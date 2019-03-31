from random import randrange
import numpy as np
import math

"""
    This program implements GDA algorithm  on Spambase Dataset
"""

class GDA:
    def fit(self, X, Y, cols, datcnt):
        """
        Generate GDA model
        :param X: train dataset
        :param Y: training labels
        :param cols:feature count
        :param datcnt: train data count
        """
        self.classes = np.unique(Y)
        tot_classes = 2
        self.phi = np.zeros((tot_classes, 1))
        self.mean = np.zeros((tot_classes, cols-1))
        self.sigma = 0
        class1, class0 = [], []
        len1= 0
        len0=0
        for i in range(len(X)):
            if Y[i] == 1.0:
                class1.append(X[i][:-1])
                len1+=1
            else:
                class0.append(X[i][:-1])
                len0+=1

        self.phi[0] = len0/(len0+len1)
        self.phi[1] = 1-self.phi[0]

        self.mean[0] = np.mean(np.array(class0), axis=0)
        self.mean[1] = np.mean(np.array(class1), axis=0)

        self.sigma = (np.cov(np.array(class0).T) * (len0 - 1) + np.cov(np.array(class1).T) * (len1 -1))/datcnt

    def predict(self, x):
        """
        Predict test labels
        :param x: test data point
        :return: prediction value for the input data point
        """
        pi=3.14

        preds=[]
        n = self.mean[0].shape[0]
        for dat in x:
            const0 = (1.0/math.sqrt(math.pow(2*pi,n) * np.linalg.det(self.sigma)))
            expterm0 = np.exp(-0.5*(dat[:-1] - self.mean[0]).T @ np.linalg.inv(self.sigma) @ (dat[:-1] - self.mean[0]))
            const1 = (1.0/math.sqrt(math.pow(2*pi,n) * np.linalg.det(self.sigma)))
            expterm1 = np.exp(-0.5*(dat[:-1] - self.mean[1]).T @ np.linalg.inv(self.sigma) @ (dat[:-1] - self.mean[1]))
            p0 = const0 * expterm0
            p1 = const1 * expterm1

            yval0 =p0* self.phi[0]
            yval1 = p1 * self.phi[1]

            if yval0[0] > yval1[0]:
                preds.append(0.0)
            else:
                preds.append(1.0)
        return preds


def read_data(filename):
    """
    Read and preprocess data from file
    :param filename: name of input file
    :return: preprocessed data
    """
    data, labels = [], []
    with open(filename, "r") as file:
        dat = file.readlines()
        for line in dat:
            newline = line.strip().replace("   ", " ").replace("  ", " ").split(',')
            data.append([float(i) for i in newline])
            # labels.append(newline[-1])
    return data


def get_normalized_data(origdata, colcount):
    """
    Normalize the data
    :param origdata: input train data
    :param colcount: number of features/columns
    :return: normalized data
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

    return origdata


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


def get_labels(x, cnt):
    """
    get labels from input dataset
    :param x: input dataset
    :param cnt: number of features/columns
    :return: label vector
    """
    lab=[]
    for d in x:
        lab.append(d[cnt-1])
    return lab

def get_accuracy(preds, actual):
    """
    Calculates accuracy of predicted labels
    :param preds: predicted labels
    :param actual: actual labels
    :return: accuracy
    """
    acc=0
    # print(actual)
    # print(len(preds), len(actual))
    if len(preds) == len(actual):
        for i in range(len(actual)):
            # print(preds[i][0], actual[i])
            if preds[i] == actual[i]:
                # print("bang")
                acc+=1
    # print(acc, len(actual))
    return acc/len(actual)


# Main processing:
k_folds = 10
data = read_data("spambase.data")
colcount = len(data[0])
norm_data = get_normalized_data(data, colcount)
# norm_data =data
# print("Data",len(norm_data))
folds = get_folds(norm_data, k_folds)
for fold in folds:
    train_data = list(folds)
    train_data.remove(fold)
    test_data = list(fold)
    train_data = sum(train_data, [])

    labels = get_labels(train_data, colcount)
    test_labels = get_labels(test_data, colcount)
    # print("test",test_data)
    # print(labels)
    model = GDA()
    GDA.fit(model, np.array(train_data), np.array(labels), colcount, len(train_data))
    preds = GDA.predict(model, test_data)
    print("Accuracy : ",get_accuracy(preds, test_labels))



