"""
prepare Training set
"""

import sys

import numpy as np
import pandas as pd

from util import PROPATH, FEAPATH, get_path, mkdir


def labs_features(fname, dire=PROPATH):
    labs = pd.read_csv(get_path(dire, 'labs.csv'))

    features = []
    with open(fname) as f:
        for line in f.readlines():
            if not line.startswith('#') and line != '\n':
                features.append(line.split()[0])

    query = labs.description == features[0]

    for f in features[1:]:
        query = query | (labs.description == f)

    return labs[query]


def fea_vitals(dire=PROPATH, save=FEAPATH):
    vitals = pd.read_csv(get_path(dire, 'vitals.csv'))
    vp = vitals.pivot(columns='measure', values='result')
    vp['id'] = vitals.id
    vp['icu'] = vitals.icu
    vp['timestamp'] = vitals.timestamp
    vp.columns = np.array(vp.columns)
    vpg = vp.groupby(['id', 'timestamp'], as_index=False).mean()

    return vpg


def fea_labs(labs):
    labs.clientresult = labs.clientresult.astype(np.float64)
    lp = labs.pivot(columns='description', values='clientresult')
    lp['id'] = labs.id
    lp['timestamp'] = labs.timestamp
    lp.columns = np.array(lp.columns)
    lpg = lp.groupby(['id', 'timestamp'], as_index=False).mean()

    return lpg


def pne(dire=PROPATH, save=FEAPATH):
    labs = labs_features('pne_feature.txt', dire)
    labs = fea_labs(labs)

    return labs


def process(dire=PROPATH, save=FEAPATH):
    mkdir(save)


if __name__ == '__main__':
    process(*sys.argv[1:])
