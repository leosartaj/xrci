"""
Utility functions
"""

import os


TRAINPATH = 'datasets/train'
PROPATH = 'datasets/pro'


def get_path(dire, fName):
    return os.path.join(os.path.expanduser(dire), fName)
