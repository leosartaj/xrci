"""
Preprocessing of data
"""

import os, sys

import numpy as np
import pandas as pd


TRAINPATH = 'datasets/train'
PROPATH = 'datasets/pro'


def _get_path(dire, fName):
    return os.path.join(os.path.expanduser(dire), fName)


def icd_9_codes(dire=TRAINPATH, save=PROPATH):
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
    assert len(names) == len(icd_9)
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
    assert len(desc) == len(icd_9)

    data = pd.DataFrame({'icd_9': icd_9, 'names': names, 'description': desc})
    if save:
        data.to_csv(_get_path(save, 'icd_9.csv'), index=False)

    return data


def pro_label(dire=TRAINPATH, save=PROPATH):
    """
    Process train_labels.csv
    Sort by Id
    Replace icd_9 codes with the appropriate names
    Replace NaN instances with -1.
    """
    labels = pd.read_csv(_get_path(dire, 'train_label.csv'))

    lsort = labels.sort_values('id_', ascending=True)
    lsort = lsort.reset_index()
    del lsort['index']

    icd_9 = pd.read_csv(_get_path(save, 'icd_9.csv'))
    assert all(lsort.columns[1:]) == all(icd_9.icd_9)
    lsort.columns = np.concatenate([['id'], icd_9.names])
    assert len(lsort.columns) == 20

    lsort = lsort.fillna(-1)
    if save:
        lsort.to_csv(_get_path(save, 'label.csv'), index=False)

    return lsort


def pro_static(dire=TRAINPATH, save=PROPATH):
    """
    Process train_Static_data.csv
    Sort by Id
    lower all the column names
    lower all the column values of gender, maritalstatus, ethnicgroup,
    admitspeciality
    """
    static = pd.read_csv(_get_path(dire, 'train_Static_data.csv'))

    static = static.sort_values('id', ascending=True)
    static = static.reset_index()
    del static['index']

    static.columns = [col.lower() for col in static.columns]
    static = static.applymap(lambda x: x.lower().replace(' ', '_').
                             replace('-', '_') if isinstance(x, str) else x)

    if save:
        static.to_csv(_get_path(save, 'static.csv'), index=False)

    return static


def pro_vitals(dire=TRAINPATH, save=PROPATH):
    """
    Process train_RawVitalData.csv
    Sort by Episode and ObservationDate
    Drop SequenceNum (not needed)
    Rename Episode to id and ObservationDate to timestamp
    lower all the column names
    lower values in measure column, replace ' ' by '_'
    """
    vitals = pd.read_csv(_get_path(dire, 'train_RawVitalData.csv'))

    vitals = vitals.sort_values(['Episode', 'ObservationDate'], ascending=True)
    vitals = vitals.reset_index()
    del vitals['index']
    del vitals['SequenceNum']

    cols = vitals.columns
    vitals.columns = np.concatenate([['id', 'timestamp'],
                                     cols[2:].map(lambda x: x.lower())])

    vitals['measure'] = vitals['measure'].map(lambda x: x.lower().
                                             replace(' ', '_'))

    if save:
        vitals.to_csv(_get_path(save, 'vitals.csv'), index=False)

    return vitals


def remove_rows(dire=TRAINPATH, save=PROPATH):
    """
    Returns the cleaned labs df and dirty rows in train_RawLabData.csv
    Computes pd.DataFrame again and again until error free.
    Very poor efficiency. Avoid it's usage.
    """
    fpath = _get_path(dire, 'train_RawLabData.csv')
    rows = []
    while True:
        try:
            labs = pd.read_csv(fpath, skiprows=rows)
            break
        except pd.parser.CParserError as e:
            line = e.args[0]
            row = line[line.index('line') + 5:line.index(',')]
            rows.append(int(row) - 1)
    if save:
        labs.to_csv(_get_path(save, 'labs_cut.csv'), index=False)

    return labs, rows


def correct_lab_data(dire=PROPATH):
    """
    Removes Irregularities in CSV train_RawLabData.csv
    call remove_rows first
    """
    files = [_get_path(dire,'labs_cut.csv'),'correctedLabData.csv']
    with open(_get_path(dire,'labs_correct.csv'),'w') as newFile:
        for fOld in files:
            with open(fOld) as Old:
                for lines in Old.readlines():
                    newFile.write(lines)


def pro_labs_basic(dire=PROPATH, save=PROPATH):
    """
    Basic Processing train_RawVitalData.csv
    Sort by Episode and ObservationDate
    Drop SequenceNum (not needed)
    lower all the column names
    lower values in columns, replace ' ' by '_'
    """
    labs = pd.read_csv(_get_path(dire, 'labs_cut.csv'))

    labs = labs.sort_values(['Episode', 'ObservationDate'], ascending=True)
    labs = labs.reset_index()
    del labs['index']
    del labs['SequenceNum']

    cols = labs.columns

    labs.columns = np.concatenate([['id', 'timestamp'],
                                     cols[2:].map(lambda x: x.lower())])

    labs['description'] = labs['description'].map(lambda x: x.lower().
                                                    replace(' ', '_'))
    labs['observationdescription'] = (labs['observationdescription'].
                                      map(lambda x: x.lower().replace(' ', '_')
                                          if not x is np.nan else x))
    labs['unitofmeasure'] = (labs['unitofmeasure'].
                                      map(lambda x: x.lower().replace(' ', '_')
                                          if not x is np.nan else x))

    labs['clientresult'] = (labs['clientresult'].
                                      map(lambda x: x.lower().replace(' ', '_')
                                          if not x is np.nan else x))

    if save:
        labs.to_csv(_get_path(save, 'labs.csv'), index=False)

    return labs


def pro_labs(dire=PROPATH, save=PROPATH):
    """
    Process train_RawVitalData.csv
    Basic processing
    Drop chstandard (62% null values)
    Drop descriptions called_to
    Correct gfr columns
    """
    labs = pro_labs_basic(dire, None)

    del labs['chstandard']

    labs = labs[(labs.description != 'called_to')]


    labs[(labs.description != 'called_to')]

    # gfr correction
    gfr = labs[labs.clientresult == '>60'].description.unique()[:-1]

    # gfr[0] and gfr[1] are very similar
    labs = labs[labs.description != gfr[1]]
    labs = labs[~((labs.description == gfr[0]) &
                  (labs.clientresult == 'canceled'))]
    #labs = labs.replace(gfr[0], 'gfr1') # renaming

    if save:
        labs.to_csv(_get_path(save, 'labs.csv'), index=False)

    return labs


def process(dire=TRAINPATH, save=PROPATH):
    """
    Preprocesses all of the data
    Generates icd_9 DataFrame
    Process train_label.csv
    """
    if not os.path.isdir(save):
        os.mkdir(save)
    icd_9_codes(dire, save)
    pro_label(dire, save)
    pro_static(dire, save)
    pro_vitals(dire, save)
    remove_rows(dire, save)
    correctLabData(save)
    pro_labs(save, save)


if __name__ == '__main__':
    process(*sys.argv[1:])
