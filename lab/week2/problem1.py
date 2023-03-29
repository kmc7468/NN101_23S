import torch
from random import random
from typing import Callable

##                         Problem 1                          ##
##                                                            ##
##            Arbitary x_train, y_train are given.            ##
##   Suppose that x and y have linear correlation, y=wx+b.    ##
##     In function training(), you should return [w, b].      ##
##          In function predict(), you should return          ##
##            list y_test corresponding to x_test.            ##
##                  Made by @jangyoujin0917                   ##
##                                                            ##

# NOTE : Feel free to use torch.optim and tensor.

def training(x_train, y_train): # DO NOT MODIFY FUNCTION NAME
    # Data normalization code (prevents overflow when calculating MSE, prevents underfitting)
    # Note that you need to convert [w, b] to the original scale before returning value
    # w = w * (y_max - y_min)
    # b = b * (y_max - y_min) + y_min
    y_min = min(y_train)
    y_max = max(y_train)
    normalize = lambda y : (y - y_min)/(y_max - y_min)

    ### IMPLEMENT FROM HERE
    y_train_normalized = [ normalize(y) for y in y_train ]

    w = torch.tensor(random(), requires_grad=True)
    b = torch.tensor(random(), requires_grad=True)
    my_lr = 0.1
    epoch = 1000

    x_train_tensor = torch.tensor(x_train, requires_grad=True)
    y_train_tensor = torch.tensor(y_train_normalized, requires_grad=True)
    optimizer = torch.optim.Adam([w, b], lr=my_lr)

    for i in range(epoch):
        optimizer.zero_grad()

        y_tensor = w * x_train_tensor + b
        error =  (y_tensor - y_train_tensor).pow(2).sum()
        error.backward()
        optimizer.step()
    
    return [ w.data.item() * (y_max - y_min), b.data.item() * (y_max - y_min) + y_min ]



def predict(x_train, y_train, x_test): # DO NOT MODIFY FUNCTION NAME
    ### IMPLEMENT FROM HERE
    w, b = training(x_train, y_train)

    temp = torch.tensor(w) * torch.tensor(x_test) + torch.tensor(b)

    return temp.tolist()

if __name__ == "__main__":
    x_train = [0.0, 1.0, 2.0, 3.0, 4.0]
    y_train = [2.0, 4.0, 6.0, 8.0, 10.0] # Note that not all test cases give clear line.
    x_test = [5.0, 10.0, 8.0]
    
    w, b = training(x_train, y_train)
    y_test = predict(x_train, y_train, x_test)

    print(w, b)
    print(y_test)
