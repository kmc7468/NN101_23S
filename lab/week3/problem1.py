import torch
import torch.nn.functional as F
from random import random 
from typing import Callable

##                         Problem 1                          ##
##                                                            ##
##            Arbitary x_train, y_train are given.            ##
##          In function predict(), you should return          ##
##            list y_test corresponding to x_test.            ##
##               y_train only contains 0 and 1.               ##
##             Therefore, use logstic regression.             ##
##                  Made by @jangyoujin0917                   ##
##                                                            ##

# NOTE : 1. Feel free to use torch.optim and tensor.
#        2. In this problem, we will only grade 'predict' function.
#           Function 'training' is only for modularization.

def training(x_train, y_train): # DO NOT MODIFY FUNCTION NAME
    x_train_tensor = torch.tensor(x_train, requires_grad=True)
    y_train_tensor = torch.tensor(y_train, requires_grad=True)
    
    w = torch.tensor([ random() for i in range(len(x_train[0])) ], requires_grad=True)
    b = torch.tensor(random(), requires_grad=True)

    my_lr = 0.01
    epoch = 10000

    optimizer = torch.optim.Adam([w, b], lr=my_lr)
    
    for i in range(epoch):
        optimizer.zero_grad()

        y_tensor = 1 / (1 + torch.exp(-(torch.matmul(w, x_train_tensor.transpose(0, 1)) + b)))
        error = (y_tensor ** (1 - y_train_tensor) + (1 - y_tensor) ** y_train_tensor).sum()
        error.backward()
        optimizer.step()

        #if i % 500 == 0:
        #    print(error)
    
    return w.tolist(), b.data.item()


def predict(x_train, y_train, x_test): # DO NOT MODIFY FUNCTION NAME
    w, b = training(x_train, y_train)
    
    return 1 / (1 + torch.exp(-(torch.matmul(torch.tensor(w), torch.tensor(x_test).transpose(0, 1)) + torch.tensor(b))))

if __name__ == "__main__":
    # This is very simple case. Passing this testcase do not mean that the code is perfect.
    # Please consider for the practial problems when score is not high.
    x_train = [[0., 1.], [1., 0.], [2., 5.], [3., 1.], [4., 2.]]
    y_train = [0., 0., 1., 0., 1.]
    x_test = [[0., 1.], [1., 0.], [2., 5.], [3., 1.], [4., 2.], [7., 2.], [1.5, 1.], [2.5, 0.5]]
    
    y_test = predict(x_train, y_train, x_test)

    print(y_test)