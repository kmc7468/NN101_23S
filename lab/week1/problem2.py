import torch
from random import random 
from typing import Callable


##                        Problem 2                           ##
##                                                            ##
##           Arbitrary quartic function will be given.        ##
## Return the optimal point(global minimum) of given function ##
##          Condition: highest order term is positive         ##
##                  Made by @jangyoujin0917                   ##
##                                                            ##


def solution(func: Callable, start_point: float) -> float: # DO NOT MODIFY FUNCTION NAME    
    my_lr = 0.8
    epoch = 45000

    x = torch.tensor(start_point, requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=my_lr, )

    for i in range(epoch):
        optimizer.zero_grad()
        y = func(x)
        y.backward()
        optimizer.step()
    
    return x.data.item()
    

if __name__ == "__main__":
    def test_func(x): # function for testing;function for evaluation will be different.
        return 4 * x ** 4 - 3 * x ** 2 - x + 1
    t = 10*random()
    print(solution(test_func, t))