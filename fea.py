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

    return vpg


def pne(dire=PROPATH, save=FEAPATH):
    pass


def process(dire=PROPATH, save=FEAPATH):
    mkdir(save)


if __name__ == '__main__':
    process(*sys.argv[1:])
