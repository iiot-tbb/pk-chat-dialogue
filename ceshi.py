#!/usr/bin/env python
# coding=utf-8
#import paddle.fluid as fluid
#import numpy as np
#
#x = np.array([[2, 2],[5,5],[6,6]], np.float32)
#with fluid.dygraph.guard(fluid.CPUPlace()):
#    inputs = []
#    a = fluid.dygraph.to_variable(x) 
#    c = (a==5)
#    print(a)
#    print(c)


class myre():
    def __init__(self):
        pass
    @classmethod
    def register(cls):
        print(cls)
    


if __name__ == "__main__":
    myre.register()
