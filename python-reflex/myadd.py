from mymul import *

def mydel(argdict):
    return 1

def myadd(argdict):
    tmp = 0
    for value in argdict.values():
        tmp += value
    print(tmp)
    tmp += mymul(argdict)
    print(tmp)
    tmp -= mydel(argdict)
    return tmp
