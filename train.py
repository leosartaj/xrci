"""
Scikit-learn ready
"""

import numpy as np
import pandas as pd
import sklearn


def oversample(df, factor=10):
    dis = df.columns[-1]
    p = df[df[dis] == 1]
    n = pd.DataFrame()
    for i in range(factor - 1):
        n = n.append(p)
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


def get_xy(df, normal=True, factor=1):
    if factor > 1:
        df = oversample(df, factor)
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
