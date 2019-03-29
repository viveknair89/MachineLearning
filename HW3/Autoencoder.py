import numpy as np


input = [[1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,], [0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1]]
fl_input=[]
for i in input:
    j=[]
    for w in i:
        j.append(float(w))
    fl_input.append(j)
# lr = 0.01
# num_steps = 2300
lr = 0.1
num_steps = 350
features= 8
hidden_units= 3
np.random.seed(30)
encoder_hidden_wt = np.random.rand(features, hidden_units)
# print("encoder:", encoder_hidden_wt)
decoder_hidden_wt = np.random.rand(hidden_units, features)

encoder_bias = np.random.rand(hidden_units)
decoder_bias = np.random.rand(features)


def sigmoid(x):
    return 1/(1+np.exp(-x))


def encoder(x):
    hidden_z = np.dot(x, encoder_hidden_wt) + encoder_bias
    layer = sigmoid(hidden_z)
    return layer


def decoder(x):
    out_z = np.dot(x, decoder_hidden_wt) + decoder_bias
    layer = sigmoid(out_z)
    return layer

loss= []
acc =[]
x_data = np.array(fl_input)
encoded = encoder(np.array(x_data))
decoded = decoder(encoded)

for i in range(num_steps):

    # Feed Forward
    encoded = encoder(np.array(x_data))
    decoded = decoder(encoded)
    hidden_z = np.dot(x_data, encoder_hidden_wt) + encoder_bias
    out_z = np.dot(decoder_hidden_wt, x_data) + decoder_bias

    # Back Propagation
    diff_cost_ow = np.dot(encoded.T, (decoded - x_data))

    diff_cost__hr = np.dot((decoded - x_data), out_z.T)

    diff_rh_zh = sigmoid(hidden_z)*(1-sigmoid(hidden_z))

    diff_cost_hw = np.dot(x_data.T, diff_rh_zh*diff_cost__hr)
    diff_cost_bh = diff_cost__hr*diff_rh_zh

    encoder_hidden_wt = encoder_hidden_wt - lr * diff_cost_hw
    encoder_bias = encoder_bias - lr * diff_cost_bh.sum(axis=0)

    decoder_hidden_wt = decoder_hidden_wt - lr* diff_cost_ow
    decoder_bias = decoder_bias - lr* (decoded - x_data).sum(axis=0)

    # accuracy = get_accuracy(output_res, y)
    # acc.append(accuracy)
    # if i %10 ==0:
    l =  ((1 / 2) * (np.power((decoded - x_data), 2)))
    # print("loss :", l.sum())
    loss.append(l)

    acc = 0
    for ind in range(len(decoded)):
        # print(input[ind], decoded[ind])
        if np.argmax(input[ind]) == np.argmax(decoded[ind]):
            acc += 1
    print("Iteration: ", i)
    print("Number of Matching codes: ", acc)
    print()
# print(decoded)
# print(l)
# print(decoded.shape)

# acc =0
# for ind in range(len(decoded)):
#     # print(input[ind], decoded[ind])
#     if np.argmax(input[ind]) == np.argmax(decoded[ind]):
#         acc+=1
#
# print("Number of Matching codes: ", acc)

def sigmoid(x):
    return 1/(1+np.exp(-x))



