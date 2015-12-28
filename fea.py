"""
prepare Training set
"""

import sys

import numpy as np
import pandas as pd

from util import PROPATH, FEAPATH, get_path, mkdir


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


def fea_vitals(dire=PROPATH, save=FEAPATH):
    vitals = pd.read_csv(get_path(dire, 'vitals.csv'))
    vp = vitals.pivot(columns='measure', values='result')
    vp['id'] = vitals.id
    vp['icu'] = vitals.icu
    vp['timestamp'] = vitals.timestamp
    vp.columns = np.array(vp.columns)
    vpg = vp.groupby(['id', 'timestamp'], as_index=False).mean()

    return vpg


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


def pne(dire=PROPATH, save=FEAPATH):
    features, ranges = get_features_ranges('pne_feature.txt')
    labs = labs_features(features, dire)
    labs = fea_labs(labs)

    return labs


def process(dire=PROPATH, save=FEAPATH):
    mkdir(save)


if __name__ == '__main__':
    process(*sys.argv[1:])
