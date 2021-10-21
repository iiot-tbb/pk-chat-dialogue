#!/usr/bin/env python
# coding=utf-8



#统计数据集中最长的句子是多长
from collections import defaultdict

def min_max_len(filename):
    min_l,max_l =float("inf"),-1
    for line in open(filename,'r'):
        lens = len(line.strip().split(" "))
        if lens>max_l: max_l = lens
        if lens<min_l: min_l = lens

    return min_l,max_l

if __name__ == "__main__":
    print("dailydialog : ",min_max_len("../data/DailyDialog/dial.train"))
    print("personchat : ", min_max_len("../data/PersonaChat/dial.train"))    
    print("DSTC : ", min_max_len("../data/DSTC7_AVSD/dial.train"))
    print("DDE : ", min_max_len("../data/DDE_Dialog/dial.train"))
    
    
