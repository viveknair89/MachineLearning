from decimal import Decimal

import numpy as np


class multinn:
    def __init__(self, x, y):
        self.input = x
        self.out = y
        self.learnrate = 0.5
        in_dim = len(x[0])
        self.size = len(x)
        classes = 3
        hidden_units = 13
        self.hidden_w = np.random.rand(in_dim, hidden_units)
        self.hidden_bias = np.random.rand(hidden_units)
        self.out_w = np.random.rand(hidden_units, classes)
        self.out_bias = np.random.rand(classes)

    def feed_fwd(self):
        hidden_z = np.dot(self.input, self.hidden_w) + self.hidden_bias

        self.hidden_res = sigmoid(hidden_z)
        output_z = np.dot(self.hidden_res, self.out_w) + self.out_bias
        self.output_res = softmax(output_z)

    def backprop(self):
        loss = calc_error(self.output_res, self.out, self.size)
        print("Error :", loss)

        out_delta = calc_cross_entropy(self.output_res, self.out, self.size)
        hidden_deltaz = np.dot(out_delta, self.out_w.T)
        hid_delta = hidden_deltaz * calc_sigmoid_der(self.hidden_res)

        self.out_w = self.out_w - self.learnrate * np.dot(self.hidden_res.T, out_delta)
        self.out_bias = self.out_bias - self.learnrate * np.sum(out_delta, axis=0, keepdims=True)
        self.hidden_w = self.hidden_w - self.learnrate * np.dot(self.input.T, hid_delta)
        self.hidden_bias = self.hidden_bias - self.learnrate * np.sum(hid_delta, axis=0)

    def get_predict(self, data):
        self.input = data
        self.feed_fwd()
        return self.output_res


def calc_error(pred, label, size):
    logp = -np.log(pred[np.arange(size), label.argmax(axis=1)])
    loss = np.sum(logp)/size
    return loss


def calc_sigmoid_der(x):
    return x * (1-x)


def calc_cross_entropy(pred, label, size):
    return (pred-label)/size


def sigmoid(x):
    # for i in x:
    #     for j in i:
    #         # j = float(Decimal(1.0)/(Decimal(1.0) + Decimal(np.exp(-j))))
    #         j= 1/(1+np.exp(-j))
    #         # j[j == 1.0] = 0.9999
    #         # j[j == 0.0] = 0.0001
    #         print(j)
    return 1/(1+ np.exp(-x))
    # return x


def softmax(x):

    # expx = np.exp(x - np.max(x, axis=1, keepdims=True))
    # return expx / expx.sum(expx, axis=1, keepdims=True)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def read_data(filename):
    data, labels = [], []
    m = {'1': [1.0, 0.0, 0.0], '2': [0.0, 1.0, 0.0], '3': [0.0, 0.0, 1.0]}
    with open(filename, "r") as file:
        dat = file.readlines()
        for line in dat:
            newline = line.strip().replace("   ", " ").replace("  ", " ").split(',')
            data.append([float(i) for i in newline[1:]])
            labels.append(m[newline[0]])
    return data, np.array(labels)


def get_normalized_data(origdata, colcount, traincount):
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

    return np.array(origdata[:traincount]), np.array(origdata[traincount:])


def get_accuracy(data, labels):
    # print("predict:",prediction[0])
    # print("labels: ", labels)
    acc = 0
    pred = model.get_predict(data)
    # print(pred, y)
    for x, y in zip(pred, labels):
        if np.argmax(x) == np.argmax(y):
            acc+=1

    return acc/len(labels)



# Main processing:

train_dat, train_labels = read_data("train_wine.csv")
test_dat, test_labels = read_data("test_wine.csv")
train_x, test_x = get_normalized_data(train_dat+test_dat, len(train_dat[0]), len(train_dat))

model = multinn(train_x, train_labels)

epochs = 1000

for x in range(epochs):
    model.feed_fwd()
    model.backprop()
    print("Iteration: ", x)
    print("Training accuracy : ", get_accuracy(train_x, train_labels))
    print("Test accuracy : ", get_accuracy(test_x, test_labels))
    print()



