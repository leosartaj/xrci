import pre
import pandas as pd
import numpy as np
labs = pre.regen_labs_data()
label = pd.read_csv('datasets/pro/label.csv')

def return_ids(dis):
    com = 'tails = label[label.'+dis+' != -1]'
    exec com
    return list(tails.id.unique())

def return_labs(ids):
    tList = []
    for i in ids:
        k = labs[labs.id == i]
        tList.append(list(k.description.unique()))
    return tList

def return_total_tests(testList):
    total = []
    for i in testList:
        for j in i:
            if j not in total:
                total.append(j)
    return total

def return_reliable(tlol,tList,thre=0):
    count = 0
    final_labs = []
    for k in tList:
        for l in tlol:
            if k in l:
                count+=1
        if count>=thre:
            final_labs.append(k)
        count = 0
    return final_labs

def main_fn(dis,thre=0):
    ids = return_ids(dis)
    labslol = return_labs(ids)
    tList = return_total_tests(labslol)
    y = return_reliable(labslol,tList,thre)
    return y
