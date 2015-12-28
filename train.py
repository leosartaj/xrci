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
    df.iloc[:, 2:-1] = df.iloc[:, 2:-1].apply(lambda x: (x - x.mean()) / x.std())
    return df


def get_xy(df, normal=True, factor=1):
    df = oversample(df, factor)
    if normal:
        df = normalize(df)
    x = np.array(df.iloc[:, 2:-1])
    y = np.array(df.iloc[:, -1])
    return x, y


def stats(y, pred):
    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    for idx, i in enumerate(y):
        if i == 0 and pred[idx] == 0:
            tn = tn + 1.0
        elif i == 0 and pred[idx] == 1:
            tp = tp + 1.0
        elif i == 1 and pred[idx] == 1:
            tp = tp + 1.0
        elif i == 1 and pred[idx] == 0:
            fn = fn + 1.0

    print 'Specificity -> %f' %(tn / (tn + fp))
    print 'Sensitivity -> %f' %(tp / (tp + fn))
