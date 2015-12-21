"""
Preprocessing of data
"""

import os

import numpy as np
import pandas as pd


TRAINPATH = 'datasets/train'
SAVEPATH = 'datasets/pro'


def _get_path(dire, fName):
    return os.path.join(os.path.expanduser(dire), fName)


def icd_9_codes(dire=TRAINPATH, save=SAVEPATH):
    """
    Creates a dataframe mapping icd_9 codes
    to names and descriptions

    returns: pd.DataFrame
    """
    labels = pd.read_csv(_get_path(dire, 'train_label.csv'))
    icd_9 = labels.columns[1:].astype(np.float64)
    names = np.array(['pne', 'cao', 'seps', 'chf', 'hfu', 'ami', 'pulm_ei',
                      'cshf', 'sseps', 'sepshock', 'cdhf', 'intes_infec',
                      'pneitus', 'dhf', 'shf', 'sub_infrac', 'bas', 'tia',
                      'ischemic_cd'])
    desc = np.array(['Pneumonia', 'Cerebral Artery Occlusion', 'Sepsis',
                     'Chronic Heart Failure', 'Heart Failure Unspecified',
                     'Acute Myocardial Infarction',
                     'Pulmonary Embolism & Infarction',
                     'Chronic systolic heart failure',
                     'Severe Sepsis', 'Septic Shock',
                     'Chronic Diastolic heart failure',
                     'Intestinal Infection due to clostridium difficile',
                      'Pneumonitus due to inhalation of food or vomitus',
                     'Diastolic Heart Failure', 'Systolic Heart Failure',
                     'Subendocardial Infarction', 'Basilar Artery Syndrome',
                     'Transient Schemic Attack',
                     'Ischemic cerebrovascular disease'])

    data = pd.DataFrame({'icd_9': icd_9, 'names': names, 'description': desc})
    data.to_csv(_get_path(save, 'icd_9.csv'), index=False)

    return pd.DataFrame({'icd_9': icd_9, 'names': names, 'description': desc})
