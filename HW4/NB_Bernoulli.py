
import math

"""
    This program implements the Naive Bayes classifier on spambase dataset with Bernoulli Distribution function
"""
def bern_predict(model, data):
    """
    Predict labels for given data
    :param model: Bernoulli model built on train data
    :param data: input test data
    :return: predicted label
    """
    prob = get_probabilities(model, data)
    best_label = None
    best_prob = -1
    # print(prob)
    for lab, pr in prob.items():
        if best_label is None or pr > best_prob:
            best_label= lab
            best_prob =pr
    return best_label


def get_probabilities(model, data):
    """
    calculate probability values for given data using trained bernoulli model
    :param model: Bernoulli model built on train data
    :param data: input test data
    :return: probability values
    """
    prob = {}
    for lab, dat in model.items():
        prob[lab]=1
        for i in range(len(dat["prob"])):
            est_prb = dat["prob"][i]
            x = data[i]
            prb = math.pow(est_prb, x)
            if prb == 1.0:
                prb = 1 - est_prb
            prob[lab] = prob[lab] * prb
        prob[lab] = prob[lab] * dat["prior_prob"]
    # print(prob)
    final_prob={}
    # print(prob.values())
    tot_prob = sum(prob.values())+ 0.00000000000000001
    for label, val in prob.items():
        final_prob[label] = val/tot_prob
    return final_prob


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
    # print( preds)
    # print(len(preds), len(actual))
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
        # print(len(line[:-1]))
        classified_data[last].append(line)
    # print(classified_data[1.0])
    return classified_data



def calc_est(dataset):
    """
    Calculated estimated label probabilities using bernoulli distribution function
    :param dataset: input dataset
    :return: estimated probability
    """
    probs = []
    for feature in zip(*dataset):
        sum_features = sum(feature)
        # if sum_features == 0.0:
        #     sum_features = 0.000000001
        probs.append((sum_features+1)/(len(dataset)+2))
    del probs[-1]
    # print(len(probs))
    return probs


def train(data, prob0, prob1):
    """
    Calculates label/class wise model for naive bayes prediction
    :param data: input data
    :param prob0: probability of class 0
    :param prob1: probability of class 1
    :return: trained naive bayes model
    """
    prob={}
    for label, dat in data.items():
        if label == 1.0:
            prob[label] = {"prior_prob": prob1, "prob": calc_est(dat)}
        else:
            prob[label] = {"prior_prob": prob0, "prob": calc_est(dat)}
    return prob



# Main processing:
k_folds = 10
data = read_data("processed.data")
colcount = len(data[0])
norm_data = get_normalized_data(data, colcount)
# norm_data =data
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

    # Classified data
    classified_data = get_data_by_class(train_data)

    # Class probabilities for train data
    prob0 = len(classified_data[0.0])/len(train_data)
    prob1 = len(classified_data[1.0])/len(train_data)
    # print(prob0, prob1)

    model = train(classified_data, prob0, prob1)

    preds=[]
    for t in test_data:
        preds.append(bern_predict(model, t))
    # print(len(summary[0.0]))
    acc= get_accuracy(preds, test_labels)
    print("Accuracy : ", acc)
    avg_accuracy+=acc
print("Average Accuracy accross all Folds: ", avg_accuracy/k_folds)




