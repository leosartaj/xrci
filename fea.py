"""
prepare Training set
"""

import sys

import numpy as np
import pandas as pd

from util import PROPATH, FEAPATH, get_path, mkdir


def fea_vitals(dire=PROPATH, save=FEAPATH):
    vitals = pd.read_csv(get_path(dire, 'vitals.csv'))
    vp = vitals.pivot(columns='measure', values='result')
    vp['id'] = vitals.id
    vp['icu'] = vitals.icu
    vp['timestamp'] = vitals.timestamp
    vp.columns = np.array(vp.columns)
    vpg = vp.groupby(['id', 'timestamp'], as_index=False).mean()

    if save:
        vpg.to_csv(get_path(save, 'vitals_fea.csv'), index=False)

    return vpg


def get_features_ranges(fname):
    features, ranges = [], {}
    with open(fname) as f:
        for line in f.readlines():
            if not line.startswith('#') and line != '\n':
                ls = line.split()
                features.append(ls[0])
                ranges[ls[0]] = float(ls[1])

    return features, ranges


def fill_normal_values(df, ranges):
    df = df.groupby('id', as_index=False).fillna(method='pad')
    return df.fillna(ranges)


def merge_df(vitals, r_vitals, labs, r_labs):
    vl = pd.merge(vitals, labs, on=['id', 'timestamp'], how='outer')
    vl = vl.sort_values(['id', 'timestamp'], ascending=True)
    vl = vl.reset_index()
    del vl['index']
    r_vitals.update(r_labs)
    vl = fill_normal_values(vl, r_vitals)

    return vl


def cummean(df):
    dfg = df.groupby('id', as_index=False)
    cc = dfg.cumcount() + 1
    cs = dfg.cumsum()

    for c in df.columns[2:]:
        df[c] = cs[c].div(cc, axis='index')

    return df


def cut_rows(df, labels):
    dis = labels.columns[-1]

    ids = labels[labels[dis] != 0].id.unique()
    ts = labels[labels[dis] != 0][dis]

    df = df[df.id.isin(ids)]

    for i, idx in enumerate(ids):
        df = df[~((df.id == idx) & (df.timestamp > ts.iloc[i]))]

    return df


def set_y(df, dis):

    length =  df.groupby('id').apply(lambda x: x.shape[0])

    l = []
    for i in length:
        l.append(np.concatenate([np.zeros(i - 1), [1]]))

    df[dis] = np.concatenate(l)
    return df


def labs_features(features, dire=PROPATH):
    labs = pd.read_csv(get_path(dire, 'labs.csv'))

    query = labs.description == features[0]

    for f in features[1:]:
        query = query | (labs.description == f)

    labs = labs[query]

    labs.clientresult = labs.clientresult.astype(np.float64)

    return labs


def fea_labs(labs):
    lp = labs.pivot(columns='description', values='clientresult')
    lp['id'] = labs.id
    lp['timestamp'] = labs.timestamp
    lp.columns = np.array(lp.columns)
    lpg = lp.groupby(['id', 'timestamp'], as_index=False).mean()

    return lpg


def get_featureset(dis, dire=PROPATH, save=FEAPATH):
    features, r_labs = get_features_ranges(dis + '_feature.txt')
    labs = labs_features(features, dire)
    labs = fea_labs(labs)

    vitals = pd.read_csv(get_path(save, 'vitals_fea.csv'))
    _, r_vitals = get_features_ranges('vitals.txt')

    labels = pd.read_csv(get_path(dire, 'label.csv'))[['id', dis]]

    vl = merge_df(vitals, r_vitals, labs, r_labs)
    vl = cut_rows(vl, labels)
    vl = cummean(vl) # can be changed
    vl = vl.reset_index()
    del vl['index']
    vl = set_y(vl, dis) # can be changed

    return vl


def process(dire=PROPATH, save=FEAPATH):
    mkdir(save)


if __name__ == '__main__':
    process(*sys.argv[1:])
