import math
from random import randrange

import numpy as np


def main():

    data= '/Users/viveknair/Desktop/ml/hw1/perceptronData.txt'
    listdata = get_data(data)

    # number of features including label
    colcount = len(listdata[0])

    # Convert feature data to float
    for i in range(colcount):
        convert_to_float(listdata, i)

    learnrate = 0.001
    print("learning rate: ", learnrate)
    # change sign for negative labels
    for dat in listdata:
        # print(dat)
        if int(dat[-1]) < 0:
            # print("less")
            dat[-1] = abs(dat[-1])
            for f in range(colcount - 1):
                dat[f] = -dat[f]

    # Get label vectors for training and test data
    labels = get_labels(listdata, colcount)

    # Add bias to data
    traindata = add_bias(listdata)


    # Remove labels from train, test data
    for dat in traindata:
        del dat[-1]

    x = np.array(traindata)

    y1 = np.array(labels)
    y = np.reshape(y1, (len(labels), 1))

    # Initialize w
    w = np.ones(colcount)

    total_mistakes = len(traindata)
    iter = 0
    while total_mistakes>0:
        iter+=1
        mistakes =0

        for data in x:
            w += learnrate*data.T
        preds = np.dot(x,w)
        missclassified = []
        predicted =preds.tolist()
        for ind in range(len(predicted)):
            if predicted[ind] < 0.0:
                missclassified.append(traindata[ind])
                mistakes+=1
        x = np.array(missclassified)
        total_mistakes= mistakes
        print("Iteration %d total mistakes: %d" % (iter, mistakes))

    w_list = w.tolist()
    print("Classifier weights: ", w_list)
    w0 = w_list[0]
    w_new = w.tolist()[1:]
    w_normalized = np.array(w_new)/w0
    print("Normalized Weights: ", w_normalized.tolist())




def error_avg(error):
    sum=0.0
    for err in error:
        sum += err
    return sum/len(error)


def get_data(filename):

    listdata = []
    with open(filename, "r") as file:
        data = file.readlines()
        for line in data:
            newline = line.strip().split('\t')
            if newline[0] != '':
                listdata.append(newline)

    return listdata


def add_bias(data):
    for i in range(len(data)):
        data[i] = [1] + data[i]
    return data


def get_labels(data,colcount):
    labels=[]
    for datapoint in data:
        labels.append(datapoint[colcount-1])
    return labels


def convert_to_float(data,feature):
    for datapoint in data:
        datapoint[feature] = float(datapoint[feature])


def get_normalized_data(origdata, colcount):
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

    return origdata





if __name__ == '__main__':
    main()