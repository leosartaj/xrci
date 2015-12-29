"""
Scikit-learn ready
"""

import numpy as np
import pandas as pd
import sklearn


def oversample(df, query=1, factor=10):
    dis = df.columns[-1]
    p = df[df[dis] == query]
    n = pd.DataFrame()
    for i in range(int(factor - 1)):
        n = n.append(p)
    rem = int((factor - int(factor)) * p.shape[0])
    if rem:
        n = n.append(p.iloc[:rem])
    return df.append(n)


def normalize(df):
    n = {}
    for c in df.columns[2:-1]:
        d = df[c]
        n[c] = (d.mean(), d.std())
        df.ix[:, c] = df.ix[:, c].apply(lambda x: (x - n[c][0]) / n[c][1])

    return df, n


def apply_normalize(df, n):
    for c in df.columns[2:-1]:
        df.ix[:, c] = df.ix[:, c].apply(lambda x: (x - n[c][0]) / n[c][1])
    return df


def get_xy(df, normal=True, factor=1, factor2=1):
    if factor > 1:
        df = oversample(df, 1, factor)
    if factor2 > 1:
        df = oversample(df, 0, factor2)
    n = {}
    if normal:
        df, n = normalize(df)
    x = np.array(df.iloc[:, 2:-1])
    y = np.array(df.iloc[:, -1])
    return x, y, n


def pred_xy(df, n=None, normal=True):
    if normal:
        df = apply_normalize(df, n)
    x = np.array(df.iloc[:, 2:-1])
    y = np.array(df.iloc[:, -1])
    return x, y
