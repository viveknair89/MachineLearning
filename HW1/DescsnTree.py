import math
import pickle
from random import randrange


def main():
    train_data = '/Users/viveknair/Desktop/ml/hw1/spambase.txt'
    listdata = get_data(train_data)

    # number of features including label
    colcount = len(listdata[0])

    # Convert each value into float
    for i in range(colcount-1):
        convert_to_float(listdata, i)

    # print(listdata)
    # ind=0
    # Get labels and minimum value for each feature for normalization
    # for datapoint in listdata:
    #     ind += 1
    #     for col in range(colcount-1):
    #         if ind == 1:
    #             minim.append(datapoint[col])
    #         else:
    #             if datapoint[col] < minim[col]:
    #                 minim[col] = datapoint[col]
    # normalizeddata = get_normalized_data(minim, listdata, colcount)
    # normalizeddata = listdata

    # Splitting dataset into k folds
    k_folds = 5
    folds = get_folds(listdata, k_folds)
    max_depth = 10
    cur_depth = 1
    min_size = 10
    labels = []
    initial_entropy = 0.0
    scores=[]

    iteration=0
    # for fold in folds:
    fold=folds[3]
    iteration += 1
    train_set = list(folds)
    train_set.remove(fold)
    test_set = fold

    # Combining k-1 training folds into single list
    train_set = sum(train_set, [])
    for t in train_set:
        labels.append(t[colcount - 1])

    total_labels = len(labels)
    label_count = get_label_count(labels)

    for l in label_count.keys():
        initial_entropy += (label_count[l] / total_labels) * (math.log2(total_labels / label_count[l]))

    # Build root node of the tree
    print(" creating root node of the tree for fold %d "% iteration)
    print("\n")
    tree_root = get_best_split(train_set, colcount, initial_entropy)

    print(" Building tree for fold %d \n"% iteration)
    print("\n")

    split_tree(tree_root, max_depth, cur_depth, min_size, colcount)

    predictions = []
    actual_labels = []
    for datapnt in test_set:
        pred = predict(tree_root, datapnt)
        predictions.append(int(pred))
        actual_labels.append(int(datapnt[colcount - 1]))

    with open('actual.pkl','wb') as f:
        pickle.dump(actual_labels, f)
    with open('predicted.pkl','wb') as s:
        pickle.dump(predictions,s)

    match = 0
    print("Labels for fold %d: " % iteration)
    print("Predicted Labels : ", predictions)
    print("Actual Labels :", actual_labels)
    # Calc accuracy
    for i in range(len(actual_labels)):
        if actual_labels[i] == predictions[i]:
            match += 1
    accuracy = match / len(actual_labels)
    scores.append(accuracy)

    confusion_matrix = create_confusion_mtrx(actual_labels, predictions)
    print(" Confusion Matrix: ", confusion_matrix)

    # print(scores)
    acc = 0.0
    for sc in scores:
        acc += sc
    print("Average Accuracy :", acc/len(scores))


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


def predict(tree, row):
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
    labels=[]
    for datapoint in data:
        labels.append(datapoint[colcount-1])
    return labels


def get_label_count(labels):
    label_count= {}
    for l in labels:
        if not label_count.__contains__(l):
            label_count[l] = 1
        else:
            label_count[l] += 1

    return label_count


def split_tree(root,max_depth, cur_depth, min_size, colcount):

    left = root["left"]
    right = root["right"]
    l_entropy = cal_entropy(left, colcount)
    r_entropy = cal_entropy(right, colcount)
    # If no split
    if not left or not right:
        root["left"] = root["right"] = create_leaf(left + right)
        return

    # Check depth

    if cur_depth >= max_depth:
        root["left"] = create_leaf(left)
        root["right"] = create_leaf(right)
        return

    # process and build children recursively
    if len(left) <= min_size:
        root['left'] = create_leaf(left)
    else:
        root["left"] = get_best_split(left, colcount, l_entropy)
        split_tree(root["left"], max_depth, cur_depth+1,min_size, colcount)

    if len(right) <= min_size:
        root['right'] = create_leaf(right)
    else:
        root["right"] = get_best_split(right,colcount, r_entropy)
        split_tree(root["right"], max_depth, cur_depth+1,min_size, colcount)


def create_leaf(leaf):
    colcount = len(leaf[0])
    predictions = get_labels(leaf, colcount)
    for i in range(len(predictions)):
        predictions[i] = float(predictions[i])
    predicted_label = max(predictions, key=predictions.count)
    return predicted_label


def create_confusion_mtrx(actual, predicted):
    true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0
    conf_matrix=[]
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            if predicted[i] == 1:
                true_positive+=1
            else:
                true_negative+=1
        else:
            if predicted[i] == 1:
                false_positive+=1
            else:
                false_negative+=1
    conf_matrix.append([true_positive, false_positive])
    conf_matrix.append([false_negative, true_negative])
    return conf_matrix


def convert_to_float(data,feature):
    for datapoint in data:
        datapoint[feature] = float(datapoint[feature])


def get_test_splits(feature,val,data):
    left = []
    right = []
    for datapoint in data:
        if datapoint[feature] < val:
            left.append(datapoint)
        else:
            right.append(datapoint)
    return left, right


def cal_avg(lbls):
    sum,avg = 0.0, 0.0
    for val in lbls:
        sum += val
    avg= sum/len(lbls)
    return avg


def cal_entropy(data, colcount):
    entropy=0.0
    lab = get_labels(data,colcount)
    tot_labels= len(lab)
    lab_count= get_label_count(lab)
    for l in lab_count.keys():
        entropy += (lab_count[l] / tot_labels) * (math.log2(tot_labels / lab_count[l]))

    return entropy


def cal_info_gain(left, right, colcount, data, parent_entropy):
    parent_size = len(data)
    lchild_size = len(left)
    rchild_size = len(right)
    l_entropy,r_entropy = 0.0, 0.0
    if lchild_size > 0:
        l_entropy= (lchild_size/parent_size) * cal_entropy(left, colcount)
    if rchild_size > 0:
        r_entropy = (rchild_size / parent_size) * cal_entropy(right, colcount)

    return parent_entropy - (l_entropy + r_entropy)


def get_best_split(data, colcount, initial_entropy):
    best_feature = colcount
    best_ig = float("-inf")
    best_val = 999
    best_grp_l, best_grp_r = None, None
    for feature in range(colcount-1):
        for datapoint in data:
            left_grp, right_grp = get_test_splits(feature, datapoint[feature], data)
            info_gain = cal_info_gain(left_grp, right_grp, colcount, data, initial_entropy)
            if info_gain > best_ig:
                best_feature = feature
                best_ig = info_gain
                best_val = datapoint[feature]
                best_grp_l = left_grp
                best_grp_r = right_grp
    return {'feature': best_feature, 'value': best_val, 'left': best_grp_l, 'right': best_grp_r}


def get_normalized_data(minim, listdata, colcount):

    maxim =[]
    datcnt =0
    for dataset in listdata:
        datcnt += 1
        for ind in range(colcount-1):
            dataset[ind]= dataset[ind]-minim[ind]
            if datcnt == 1:
                maxim.append(dataset[ind])
            else:
                if dataset[ind] > maxim[ind]:
                    maxim[ind] = dataset[ind]

    for datapoint in listdata:
        for ind in range(colcount-1):
            datapoint[ind] = datapoint[ind]/maxim[ind]

    return listdata


def get_data(filename):

    listdata =[]
    with open(filename, "r") as file:
        data = file.readlines()
        for line in data:
            newline=line.strip().replace("   ", " ").replace("  ", " ").split(',')
            if newline[0] != '':
                listdata.append(newline)

    return listdata




if __name__ == '__main__':
    main()