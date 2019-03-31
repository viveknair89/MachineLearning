

import numpy as np

"""
    This program implements the Single layered Perceptron algorithm 
"""
def main():

    # Fetch Training and testing dataset
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

    # Train until we get perfect prediction
    while total_mistakes>0:
        iter+=1
        mistakes =0

        for data in x:
            w += learnrate*data.T
        preds = np.dot(x,w)
        missclassified = []
        predicted =preds.tolist()

        # All the labels having negative labels after training would be misclassified
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
            newline = line.strip().split('\t')
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