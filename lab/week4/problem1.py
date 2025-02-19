import torch
import torch.nn.functional as F
import numpy as np
from random import random

##                         Problem 1                          ##
##                                                            ##
##           Given Iris_train.csv, train your model           ##
##              to predict the species of iris.               ##
##            In the implementation of functions,             ##
##                  opening file is invalid.                  ##
## Checker will provide Iris_train.csv with x_train, y_train. ##
## LIMITS : Do not reference dataset on Internet in the code. ##
##        (Since test datas are also on the Internet.)        ##
##                  Made by @jangyoujin0917                   ##
##                                                            ##

# NOTE : Data normalization may help you to get good score.

iris = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
def iris_onehot(x : str) -> list[int]:
    index = iris.index(x)
    return list(np.eye(len(iris))[index])

def training(x_train : torch.Tensor, y_train : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: # DO NOT MODIFY FUNCTION NAME
    w = torch.tensor([ [ random() for j in range(4) ] for i in range(3) ], requires_grad=True)
    b = torch.tensor([ [ random() ] for i in range(3) ], requires_grad=True)
    o = torch.tensor([ [ 1.0 ] for i in range(x_train.shape[0]) ], requires_grad=True).transpose(0, 1)

    my_lr = 0.01
    epoch = 10000

    optimizer = torch.optim.Adam([w, b], lr=my_lr)
    loss = torch.nn.CrossEntropyLoss()

    for i in range(epoch):
        optimizer.zero_grad()

        y = torch.matmul(w, x_train.transpose(0, 1)) + torch.matmul(b, o)

        error = loss(y.transpose(0, 1), y_train)
        error.backward()
        optimizer.step()

        #if i % 500 == 0:
        #    print(error)

    return w, b
        

def predict(x_train : torch.Tensor, y_train : torch.Tensor, x_test : torch.Tensor) -> torch.Tensor: # DO NOT MODIFY FUNCTION NAME
    # predict() should return the index of the answer.
    # Therefore, the return value is 1d-tensor that only contains 0, 1, 2. 
    w, b = training(x_train, y_train)
    
    o = torch.tensor([ [ 1.0 ] for i in range(x_test.shape[0]) ]).transpose(0, 1)
    y = (torch.matmul(w, x_test.transpose(0, 1)) + torch.matmul(b, o)).transpose(0, 1)

    sm = torch.nn.Softmax()
    y = sm(y)
    
    return torch.argmax(y, dim=1)

if __name__ == "__main__":
    import csv
    with open("Iris_train.csv", "r") as f: # Make sure that Iris_train.csv is in same directory.
        rdr = csv.reader(f)
        contents = [i for i in rdr]
    contents = contents[1:]
    x = torch.tensor([[float(j) for j in i[:-1]] for i in contents], dtype=torch.float32)
    y = torch.tensor([iris_onehot(i[-1]) for i in contents], dtype=torch.float32)
    test = torch.tensor([[5.1,3.0,1.2,0.3], [5.1,3.4,4.1,1.3]], dtype=torch.float32)

    prediction = predict(x, y, test)
    print(prediction)