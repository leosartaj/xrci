"""
Scikit-learn ready
"""

import numpy as np
import pandas as pd
from sklearn.externals import joblib

from util_my import get_path, mkdir, MODELPATH


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


def split_df(df, factor=.9, factor2=.9):
    dis = df.columns[-1]
    labels = pd.read_csv('datasets/pro/label.csv')

    dis_ids = df[df[dis] == 1].id.unique()
    non_ids = labels[labels[dis] == 0].id.unique()

    dis_index = int(dis_ids.shape[0] * factor)
    non_index = int(non_ids.shape[0] * factor2)

    train_ids = np.concatenate([dis_ids[:dis_index], non_ids[:non_index]])
    valid_ids = np.concatenate([dis_ids[dis_index:], non_ids[non_index:]])

    return df[df.id.isin(train_ids)], df[df.id.isin(valid_ids)]


def save_model(model, fname, nor, nor_fname, save=MODELPATH):
    mkdir(save)
    joblib.dump(model, get_path(save, fname))
    joblib.dump(nor, get_path(save, nor_fname))


def get_model(fname, nor_fname, dire=MODELPATH):
    model = joblib.load(get_path(dire, fname))
    nor = joblib.load(get_path(dire, nor_fname))
    return model, nor
