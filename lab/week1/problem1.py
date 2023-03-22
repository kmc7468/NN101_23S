import torch
from random import random 
from typing import Callable


##                        Problem 1                           ##
##                                                            ##
##         Arbitrary quadratic function will be given.        ##
## Return the optimal point(global minimum) of given function ##
##          Condition: highest order term is positive         ##
##                  Made by @jangyoujin0917                   ##
##                                                            ##


def solution(func: Callable, start_point: float) -> float: # DO NOT MODIFY FUNCTION NAME    
    lr = 0.001
    epoch = 10000
    beta = 0.9

    x = start_point
    v = 0

    for i in range(epoch):
        x_tensor = torch.tensor([x + beta * v], requires_grad=True)
        y = func(x_tensor)
        y.backward()

        v = beta * v - lr * x_tensor.grad.data.item()
        x += v
    
    return x


if __name__ == '__main__':
    def test_func(x): # function for testing;function for evaluation will be different.
        return x ** 2
    t = 10*random()
    print(solution(test_func, t))