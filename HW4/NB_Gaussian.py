
import numpy as np
import math

"""
    This program implements the Naive Bayes Classifier on Spambase Dataset with Gaussian distribution function
"""
def predict(summary, data):
    """
    Predicts label for the input dataset using summary of trained model
    :param summary: trained model
    :param data: input data point
    :return: predicted label
    """
    prob = get_probabilities(summary, data)
    best_label = None
    best_prob = -1
    # print(prob)
    for lab, pr in prob.items():
        if best_label is None or pr > best_prob:
            best_label= lab
            best_prob =pr
    return best_label


def get_probabilities(summary, data):
    """
    Fetch probability values for each feature in data point using trained model
    :param summary: trained model
    :param data: input data point
    :return: probability values
    """
    prob = {}
    for lab, dat in summary.items():
        prob[lab]=1
        for i in range(len(dat["summary"])):
            meanval, stddev = dat["summary"][i]
            x = data[i]
            prob[lab] = prob[lab] * calc_prob(x, meanval, stddev)
        prob[lab] = prob[lab] * dat["prior_prob"]
    final_prob={}
    print(prob)
    tot_prob = sum(prob.values())+ 0.00000000000000001
    for label, val in prob.items():
        final_prob[label] = val/tot_prob
    return final_prob


def calc_prob(x, mean, stddev):
    """
    probability calculation using Gaussian function
    :param x: feature value
    :param mean: mean of feature
    :param stddev: stddev of feature
    :return: probability value of the feature
    """
    pi=3.14
    # print(stddev)
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stddev, 2))))
    return (1 / (math.sqrt(2 * pi) * stddev)) * exponent


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
    fold=[]
    for i in range(k):
        ind = i
        fold.append([])
        count=0
        while ind < len(data):
            count+=1
            fold[i].append(data[ind])
            ind+=k
    return fold

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
    if len(preds) == len(actual):
        for i in range(len(actual)):
            # print(preds[i][0], actual[i])
            if preds[i] == actual[i]:
                # print("bang")
                acc+=1
    # print(acc, len(actual))
    return acc/len(actual)


def get_data_by_class(data):
    """
    Classifies data by labels/classes
    :param data: unclassified raw  data
    :return: classified data(a dictionary with classes/labels as keys)
    """
    classified_data ={1.0: [], 0.0: []}
    for line in data:
        last = line[-1]
        classified_data[last].append(line)
    return classified_data

def calc_mean(feature):
    """
    calculates mean of the input feature vector
    :param feature: feature/column vector
    :return: mean of feature
    """
    return float(sum(feature)/len(feature))


def calc_stddev(feature):
    """
    calculates standard deviation of the input feature vector
    :param feature: feature/column vector
    :return: standard deviation of feature
    """
    avg = np.mean(feature)
    cum_diff = 0.00000000000000001
    # cum_diff = 0.0
    for x in feature:
        cum_diff += pow(x-avg, 2)
    var = cum_diff/float(len(feature)-1)
    # print("var:",var)
    return math.sqrt(var)

def calc_summary(dataset):
    """
    Calculates mean and standard deviation for each feature in dataset
    :param dataset: input dataset
    :return: gaussian model for naive bayes prediction
    """
    summarries =[]
    for feature in zip(*dataset):
        meanval = calc_mean(feature)
        stddev = calc_stddev(feature)
        summarries.append((meanval, stddev))
    del summarries[-1]

    return summarries

def get_summaries(data, prob0, prob1):
    """
    Calculates label/class wise summary/ model for naive bayes prediction
    :param data: input data
    :param prob0: probability of class 0
    :param prob1: probability of class 1
    :return: trained naive bayes model
    """
    summary={}
    for label, dat in data.items():
        if label == 1.0:
            summary[label] = {"prior_prob": prob1, "summary": calc_summary(dat)}
        else:
            summary[label] = {"prior_prob": prob0, "summary": calc_summary(dat)}

    return summary

# Main processing:
k_folds = 10
data = read_data("processed.data")
colcount = len(data[0])
# norm_data = get_normalized_data(data, colcount)
norm_data =data
# print("Data",len(norm_data))
folds = get_folds(norm_data, k_folds)
avg_accuracy=0.0
for fold in folds:
    train_data = list(folds)
    train_data.remove(fold)
    test_data = list(fold)
    train_data = sum(train_data, [])
    train_labels = get_labels(train_data, colcount)
    test_labels = get_labels(test_data, colcount)

    # Get Classified data
    classified_data = get_data_by_class(train_data)

    # Calculate probability for each label/class
    prob0 = len(classified_data[0.0])/len(train_data)
    prob1 = len(classified_data[1.0])/len(train_data)
    # print(prob0, prob1)

    # calculate class wise summary for each feature
    summary = get_summaries(classified_data, prob0, prob1)

    # Predict labels for test data
    preds=[]
    for t in test_data:
        preds.append(predict(summary, t))
    acc= get_accuracy(preds, test_labels)
    print("Accuracy : ", acc)
    avg_accuracy+=acc
print("Average Accuracy accross all Folds: ", avg_accuracy/k_folds)
    # print("tr Accuracy : ",get_accuracy(preds, train_labels))

    # get_summaries_byClass(train_data)
    # for att in zip(*train_data):

    # print("test",len(test_data))
    # print("train", len(train_data))
#     model = GDA()
#     GDA.fit(model, np.array(train_data), np.array(labels), colcount, len(train_data))
#     preds = GDA.predict(model, test_data)



