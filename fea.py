"""
prepare Training set
"""

import sys

import numpy as np
import pandas as pd

from util import PROPATH, FEAPATH, get_path, mkdir


def gen_vitals(dire=PROPATH, save=FEAPATH):
    vitals = pd.read_csv(get_path(dire, 'vitals.csv'))
    vp = vitals.pivot(columns='measure', values='result')
    vp['id'] = vitals.id
    vp['icu'] = vitals.icu
    vp['timestamp'] = vitals.timestamp
    vp.columns = np.array(vp.columns)
    vpg = vp.groupby(['id', 'timestamp'], as_index=False).mean()

    if save:
        vpg.to_csv(get_path(save, 'vitals.csv'), index=False)

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


def cummean(df, cols):
    dfg = df.groupby('id', as_index=False)
    cc = dfg.cumcount() + 1
    cs = dfg.cumsum()

    for c in cols:
        df[c] = cs[c].div(cc, axis='index')

    return df


def set_y(df, labels):
    dis = labels.columns[-1]

    ids = df.id.unique()
    length = df.groupby('id').apply(lambda x: x.shape[0])

    y = []
    for i, l in enumerate(length):
        dia = labels[labels.id == ids[i]][dis].iloc[0]
        if dia:
            s = (df[df.id == ids[i]].timestamp < dia).sum()
            if s == l:
                s = s - 1
        else:
            s = l
        y.append(np.concatenate([np.zeros(s), np.zeros(l - s) + 1]))

    df[dis] = np.concatenate(y)

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


def _get_feature(dis, dire=PROPATH, save=FEAPATH):
    features, r_labs = get_features_ranges(dis + '_feature.txt')
    labs = labs_features(features, dire)
    labs = fea_labs(labs)

    vitals = pd.read_csv(get_path(save, 'vitals.csv'))
    del vitals['icu']
    features_v, r_vitals = get_features_ranges('vitals.txt')

    vl = merge_df(vitals, r_vitals, labs, r_labs)

    cols = list(features)
    cols.extend(features_v)
    vl = cummean(vl, cols) # can be changed

    static = pd.read_csv(get_path(dire, 'static.csv'))[['id', 'age', 'gender']]
    vl = pd.merge(vl, static, on='id')

    return vl


def cut_rows_time_cut(df, labels):
    dis = labels.columns[-1]

    ids = labels[labels[dis] != 0].id.unique()
    ts = labels[labels[dis] != 0][dis]

    df = df[df.id.isin(ids)]

    for i, idx in enumerate(ids):
        df = df[~((df.id == idx) & (df.timestamp > ts.iloc[i]))]

    return df


def cut_rows_disease_cut(df, labels):
    dis = labels.columns[-1]
    ids = labels[labels[dis] != 0].id.unique()
    df = df[df.id.isin(ids)]
    return df


def get_feature_set_time_cut(dis, dire=PROPATH, save=FEAPATH):
    vl = _get_feature(dis, dire, save)

    labels = pd.read_csv(get_path(dire, 'label.csv'))[['id', dis]]
    vl = cut_rows_time_cut(vl, labels)
    vl = vl.reset_index()
    del vl['index']

    vl = set_y(vl, labels)

    if save:
        vl.to_csv(get_path(save, dis + '_feature.csv'), index=False)

    return vl


def get_feature_set_disease_cut(dis, dire=PROPATH, save=FEAPATH):
    vl = _get_feature(dis, dire, save)

    labels = pd.read_csv(get_path(dire, 'label.csv'))[['id', dis]]
    vl = cut_rows_disease_cut(vl, labels)
    vl = vl.reset_index()
    del vl['index']

    vl = set_y(vl, labels)

    if save:
        vl.to_csv(get_path(save, dis + '_feature_disease.csv'), index=False)

    return vl


def get_feature_set_complete(dis, dire=PROPATH, save=FEAPATH):
    vl = _get_feature(dis, dire, save)

    labels = pd.read_csv(get_path(dire, 'label.csv'))[['id', dis]]
    vl = set_y(vl, labels)

    if save:
        vl.to_csv(get_path(save, dis + '_feature_complete.csv'), index=False)

    return vl


def process(dire=PROPATH, save=FEAPATH):
    mkdir(save)
    gen_vitals(dire, save)
    get_feature_set_time_cut('pne', dire, save)
    get_feature_set_time_cut('cao', dire, save)
    get_feature_set_time_cut('ami', dire, save)


if __name__ == '__main__':
    process(*sys.argv[1:])
