import math

"""
    This program builds a regression tree training model on Housing Data
"""
def main():
    max_depth = 2
    cur_depth = 1
    min_size = 10

    # Fetch preprocessed training and test data
    train_data = '/Users/viveknair/Desktop/ml/hw1/housing_train.txt'
    listdata = get_data(train_data)
    test_data = '/Users/viveknair/Desktop/ml/hw1/housing_test.txt'
    testdata = get_data(test_data)

    # number of features including label
    colcount = len(listdata[0])

    # Convert feature data to float
    for i in range(colcount):
        convert_to_float(listdata, i)
        convert_to_float(testdata, i)

    # Create root node of the regression tree
    tree_root = get_best_split(listdata, colcount)


    # split the tree root into further branches recursively
    split_tree(tree_root, max_depth, cur_depth, min_size, colcount)


    # Calculate Training Error
    training_predictions = []
    training_labels = []
    for datapnt in listdata:
        pred = predict(tree_root, datapnt)
        training_predictions.append(pred)
        training_labels.append(datapnt[colcount-1])
    print("Training Predictions: ", training_predictions)
    print("Training Labels: ", training_labels)
    acc = 0.0

    # Calc mse
    for i in range(len(training_labels)):
        acc += math.pow((training_predictions[i] - training_labels[i]), 2)
    mse = acc/len(training_labels)
    print("Training Error: ", mse)
    print("\n")

    # Testing Error
    mse=0.0
    testing_predictions = []
    testing_labels = []
    for datapnt in testdata:
        pred = predict(tree_root, datapnt)
        testing_predictions.append(pred)
        testing_labels.append(datapnt[colcount - 1])
    print("Training Predictions: ", testing_predictions)
    print("Training Labels: ", testing_labels)
    acc = 0.0

    # Calc mse
    for i in range(len(testing_labels)):
        acc += math.pow((testing_predictions[i] - testing_labels[i]), 2)
    mse = acc / len(testing_labels)
    print("Training Error: ", mse)



def predict(tree, row):
    """
    Predicts test labels
    :param tree: regression tree modeled on train data
    :param row: one dataset from test set
    :return: returns leaf node referring to one of the possible output labels (0/1)
    """
    # print(tree['value'], row)
    if row[tree['feature']] < tree['value']:
        if isinstance(tree['left'], dict):
            return predict(tree['left'], row)
        else:
            return tree['left']
    else:
        if isinstance(tree['right'], dict):
            return predict(tree['right'], row)
        else:
            return tree['right']

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


def split_tree(root,max_depth, cur_depth, min_size, colcount):
    """
    Splits the tree into further branches based on the root node provided
    :param root: Parent node
    :param max_depth: maximum depth of the tree
    :param cur_depth: Current depth of the tree
    :param min_size: Minimum size limit of the branch
    :param colcount: Number of features/columns
    """
    left = root['left']
    right = root['right']
    del(root['right'])
    del(root['left'])
    # If no split
    if not left or not right:
        root['left'] = root['right'] = create_leaf(left + right)
        return

    #Check depth

    if cur_depth >= max_depth:
        root['left'] = create_leaf(left)
        root['right'] = create_leaf(right)
        return

    # process and build children recursively
    if len(left) <= min_size:
        root['left'] = create_leaf(left)
    else:
        root['left'] = get_best_split(left, colcount)
        split_tree(root['left'], max_depth, cur_depth+1, min_size, colcount)

    if len(right) <= min_size:
        root['right'] = create_leaf(right)
    else:
        root['right'] = get_best_split(right, colcount)
        split_tree(root['right'], max_depth, cur_depth+1, min_size, colcount)


def create_leaf(leaf):
    """
    Create Leaf Node with corresponding label
    :param leaf: Node which has to be made as the leaf node
    :return: Returns the label assigned to the leaf
    """
    sum = 0.0
    for row in leaf:
        sum += row[-1]
    return sum/len(leaf)


def convert_to_float(data,feature):
    """
    Converts data to float
    :param data: input data
    :param feature: number of features
    """
    for datapoint in data:
        datapoint[feature] = float(datapoint[feature])


def get_test_splits(feature,val,data):
    """
    Returns test split by splitting given data on given value
    :param feature: feature/column count
    :param val: Value on which data needs to be split
    :param data: input data
    :return: left and right child after split
    """
    left = []
    right = []
    for datapoint in data:
        if datapoint[feature] < val:
            left.append(datapoint)
        else:
            right.append(datapoint)

    return right, left

def cal_avg(lbls):
    """
    Calculates average of given labels
    :param lbls: Label values
    :return: Average of the list
    """
    sum,avg = 0.0, 0.0
    for val in lbls:
        sum += val
    avg= sum/len(lbls)
    return avg


def get_ms_err(avg, labels):
    """
    Calculate mean square error
    :param avg: average score
    :param labels: labels vector
    :return: mean square error value
    """
    mserr=0.0
    for l in labels:
        mserr += math.pow(l - avg, 2)
    return mserr


def cal_ms_err(left,right,colcount):
    """
    Compute mean square error for whole node
    :param left: left child
    :param right: right child
    :param colcount: column/feature count
    :return: mean square error for the whole node
    """
    l_avg, r_avg, l_ms_err, r_ms_err = 0.0, 0.0, 0.0, 0.0
    if len(left) > 0:
        l_lbls = get_labels(left,colcount)
        l_avg = cal_avg(l_lbls)
        l_ms_err = get_ms_err(l_avg, l_lbls)
    if len(right) > 0:
        r_lbls = get_labels(right, colcount)
        r_avg = cal_avg(r_lbls)
        r_ms_err = get_ms_err(r_avg, r_lbls)

    return l_ms_err + r_ms_err


def get_best_split(data, colcount):
    """
    Get best split with maximum information gain
    :param data: input data
    :param colcount: count of features/columns
    :param initial_entropy: Parent entropy
    :return: Best split values
    """
    best_err = math.inf
    best_feature = 999
    best_val = 9999
    best_grp_l, best_grp_r = None, None
    for feature in range(colcount-1):
        for datapoint in data:
            right_grp, left_grp = get_test_splits(feature, datapoint[feature], data)
            ms_err = cal_ms_err(left_grp, right_grp, colcount)
            if ms_err < best_err:
                best_feature = feature
                best_err = ms_err
                best_val = datapoint[feature]
                best_grp_l = left_grp
                best_grp_r = right_grp
    return {'feature': best_feature, 'value': best_val, 'left': best_grp_l, 'right': best_grp_r}


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


if __name__ == '__main__':
    main()