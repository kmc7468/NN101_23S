import torch
from random import random 
from typing import Callable

##                         Problem 2                          ##
##                                                            ##
##            Arbitary x_train, y_train are given.            ##
##     In this problem, x_train is list of list of float.     ##
##   Suppose that x and y have linear correlation, y=wx+b.    ##
##           (In this problem, w will be a vector.)           ##
##     In function training(), you should return [w, b].      ##
##          In function predict(), you should return          ##
##            list y_test corresponding to x_test.            ##
##                  Made by @jangyoujin0917                   ##
##                                                            ##

# NOTE : Feel free to use torch.optim and tensor.

def training(x_train, y_train): # DO NOT MODIFY FUNCTION NAME
    # data normalization
    # 1. Prevents overflow when calculating MSE
    # 2. Prevents underfitting
    # Note that you need to convert [w, b] to the original scale.
    # w = w * (y_max - y_min)
    # b = b * (y_max - y_min) + y_min
    y_min = min(y_train)
    y_max = max(y_train)
    normalize = lambda y : (y - y_min)/(y_max - y_min)
    
    ### IMPLEMENT FROM HERE
    y_train_normalized = [ normalize(y) for y in y_train ]

    x_train_tensors = [ torch.tensor(x, requires_grad=True) for x in x_train ]
    y_train_tensors = [ torch.tensor(y, requires_grad=True) for y in y_train_normalized ]

    w = torch.tensor([ random() for i in range(len(x_train[0])) ], requires_grad=True)
    b = torch.tensor(random(), requires_grad=True)
    my_lr = 0.1
    epoch = 70

    optimizer = torch.optim.Adam([w, b], lr=my_lr)

    for i in range(epoch):
        #optimizer.zero_grad()
        #error_mean = torch.tensor(0., requires_grad=True)

        for j in range(len(x_train)):
            optimizer.zero_grad()

            y_tensor = torch.dot(w, x_train_tensors[j]) + b

            error = (y_tensor - y_train_tensors[j]).pow(2).sum()
            error.backward()
            optimizer.step()

            #error_mean2 = error_mean + error
            #error_mean = error_mean2

        #error_mean.backward()
        #optimizer.step()

    return ((w * (y_max - y_min)).tolist(), (b * (y_max - y_min) + y_min).data.item())


def predict(x_train, y_train, x_test): # DO NOT MODIFY FUNCTION NAME
    ### IMPLEMENT FROM HERE
    w, b = training(x_train, y_train)

    result = []

    for i in range(len(x_test)):
        temp = torch.dot(torch.tensor(w), torch.tensor(x_test[i])) + torch.tensor(b)
        result.append(temp.data.item())

    return result

if __name__ == "__main__":
    x_train = [[0., 1.], [1., 0.], [2., 2.], [3., 1.], [4., 3.]]
    y_train = [3., 2., 7., 6., 11.] # y = x_0 + 2*x_1 + 1 # Note that not all test cases give clear line.
    x_test = [[5., 3.], [10., 6.], [8., 9.]]
    
    w, b = training(x_train, y_train)
    y_test = predict(x_train, y_train, x_test)

    print(w, b)
    print(y_test)