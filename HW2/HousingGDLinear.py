import numpy as np

"""
    This Program implements Linear Regression algorithm with Gradient Descent on Housing Data
"""
def main():

    # Fetch Train/Test Data and preprocess it
    train_data= '/Users/viveknair/Desktop/ml/hw1/housing_train.txt'
    listdata = get_data(train_data)
    test_data = '/Users/viveknair/Desktop/ml/hw1/housing_test.txt'
    testdata = get_data(test_data)

    # Iterations and learning rate
    epoch = 1000
    lam = 0.001

    # number of features including label
    colcount = len(listdata[0])

    # Convert feature data to float
    for i in range(colcount):
        convert_to_float(listdata, i)
        convert_to_float(testdata, i)

    print('length of train before', len(listdata))
    print('length of test before', len(testdata))
    listdata, testdata = get_normalized_data(listdata+testdata, colcount, len(listdata))

    print('length of train ',listdata[0])
    print('length of test ',testdata[0])
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


    x = np.array(listdata)
    x_test =np.array(testdata)
    w = np.zeros((colcount, 1))
    y1 =np.array(labels)
    y = np.reshape(y1, (len(labels), 1))
    y2 = np.array(labels_test)
    y_test = np.reshape(y2, (len(labels_test), 1))
    w_hist =[]

    # Calculate Gradient
    for iter in range(epoch):
        print("iteration Number: ", iter)
        prediction = np.dot(x, w)
        print(w.tolist())
        w = w - lam * np.dot(x.T, (prediction-y))

        w_hist.append(w)

    # Training error calculation
    predicted_y = np.dot(x, w)
    predicted = predicted_y.tolist()
    actual = y.tolist()
    print("predicted training label", predicted)
    print("actual training label", actual)
    acc = 0.0
    for i in range(len(actual)):
        acc += (predicted[i][0] - actual[i][0]) ** 2
    mse_train = acc / len(actual)

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
        acc1 += (predicted_test[i][0] - actual_test[i][0]) ** 2
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


def get_normalized_data(origdata, colcount, traincount):
    """
    Normalize the data
    :param traincount: number of rows in training data
    :param origdata: input data
    :param colcount: number of features/columns
    :return: normalized data(train, test)
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





if __name__ == '__main__':
    main()