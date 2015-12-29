"""
Utility functions
"""

import os


TRAINPATH = 'datasets/train'
PROPATH = 'datasets/pro'
FEAPATH = 'datasets/fea'
TESTPATH = 'datasets/test'


def get_path(dire, fName):
    return os.path.join(os.path.expanduser(dire), fName)


def mkdir(dire):
    if not os.path.isdir(dire):
        os.mkdir(dire)
