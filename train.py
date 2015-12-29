"""
Scikit-learn ready
"""

import numpy as np
import pandas as pd
import sklearn


def oversample(df, factor=1, factor2=1):
    labels = pd.read_csv('datasets/pro/label.csv')
    dis = df.columns[-1]
    n = pd.DataFrame()

    p = df[df[dis] == 1]
    for i in range(int(factor - 1)):
        n = n.append(p)

    ids = labels[labels[dis] == 0].id.unique()
    p = df[df.id.isin(ids)]
    for i in range(int(factor2 - 1)):
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


def get_xy(df, normal=True, factor=1, factor2=1):
    if factor > 1 or factor2 > 1:
        df = oversample(df, factor, factor2)
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
