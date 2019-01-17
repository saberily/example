#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#被反射的文件最好和reflex在同一级目录，这样文件的所作目录保持一致import时不会出问题

def reflex(filename, funcname, argdict):
    module = __import__(filename, fromlist=True)
    func = getattr(module, funcname, None)
    if None != func:
        return func(argdict)

argdict = {'x':1, 'y':2}
print(argdict)
print(reflex("myadd", "myadd", argdict))