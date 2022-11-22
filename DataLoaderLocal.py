import numpy as np

def r2c_7(a):
    if a < -2:
        res = 0
    elif a < -1:
        res = 1
    elif a < 0:
        res = 2
    elif a <= 0:
        res = 3
    elif a <= 1:
        res = 4
    elif a <= 2:
        res = 5
    elif a > 2:
        res = 6
    else:
        print('result can not be transferred to 7-class label', a)
        raise NotImplementedError
    return res

def mosi_r2c_7(a):
    return np.int64(np.round(a)) + 3

def r2c_2(a):
    if a < 0:
        res = 0
    else:
        res = 1
    return res

def pom_r2c_7(a):
    # [1,7] => 7-class
    if a < 2:
        res = -3
    if 2 <= a and a < 3:
        res = -2
    if 3 <= a and a < 4:
        res = -1
    if 4 <= a and a < 5:
        res = 0
    if 5 <= a and a < 6:
        res = 1
    if 6 <= a and a < 7:
        res = 2
    if a >= 7:
        res = 3
    return res + 3
