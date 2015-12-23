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
    labs = pd.read_csv(_get_path(dire, 'labs_correct.csv'))

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


def fill_see_below(labs, desc):
    """
    fills the see_below clientresults
    uses bfill method
    bfill's only for same descriptions
    """
    see = labs.clientresult == 'see_below'
    if type(desc) == str:
        desc = labs.description == desc
    ids = labs[(desc) & (see)].id.unique()
    for i in ids:
        labs.ix[(desc) & (labs.id == i) & (see), 'clientresult'] = np.nan
        labs.ix[(desc) & (labs.id == i), 'clientresult'] = labs.ix[(desc) & (labs.id == i), 'clientresult'].fillna(method='bfill')

    return labs


def _remove_desc(labs):
    """
    Removes descriptions not required from labs
    """
    labs = labs[(labs.description != 'called_to')]
    labs = labs[(labs.description != 'influenza_type_b')]

    return labs


def _pro_gfr(labs):
    """
    gfr correction
    estimated_gfr-african_american and estimated_gfr-other are very similar
    egfr_non-african_american and egfr_african_american are very similar
    rename the other two as gfr
    clean up values in clientresult
    """
    gfr = labs[labs.clientresult == '>60'].description.unique()[:-1]

    labs = labs[labs.description != gfr[1]]

    labs = labs[labs.description != gfr[3]]

    labs.ix[(labs.description == gfr[0]) | (labs.description == gfr[2]),
            'description'] = 'gfr'

    labs.ix[(labs.description == 'gfr') &
            ((labs.clientresult == '>60') |
             (labs.clientresult == '>60.00')), 'clientresult'] = 75

    return labs


def _pro_allen(labs):
    """
    Allen's Test Correction
    Results are Pass(1), Fail(0) or Half Passed 0.5 (fail = 0,else =1)
    Clean clientresult for allen's test
    """
    labs.ix[((labs.description == "allen's_test") & (labs.clientresult == "collected_by_respiratory.")), 'clientresult'] = 1
    labs.ix[((labs.description == "allen's_test") & (labs.clientresult.isnull())), 'clientresult'] = 1
    labs.ix[((labs.description == "allen's_test") & (labs.clientresult == "na")), 'clientresult'] = 1
    labs.ix[((labs.description == "allen's_test") & (labs.clientresult == ".")), 'clientresult'] = 1
    labs.ix[((labs.description == "allen's_test") & (labs.clientresult == "passed_left_radial")), 'clientresult'] = 0.5
    labs.ix[((labs.description == "allen's_test") & (labs.clientresult == "pass")), 'clientresult'] = 1
    labs.ix[((labs.description == "allen's_test") & (labs.clientresult == "fail")), 'clientresult'] = 0

    return labs


def _pro_albumin(labs):
    """
    albumin correction
    Range 3.5-5.5 g/dL (35-55 g/L) is normal
    Clean clientresult
    """
    labs.ix[(labs.clientresult == '4.0_g/dl'), 'clientresult'] = 4.0
    labs.ix[(labs.clientresult == '>_3.2_-_normal'), 'clientresult'] = 4.0
    labs.ix[(labs.clientresult == '<1.5'), 'clientresult'] = 1.5
    return labs


def _pro_alp(labs):
    """
    alkaline_phosphatase correction also called ALP or ALKP
    Rename to alp
    All units are in u/l
    """
    labs.ix[(labs.description == 'alkaline_phosphatase'), 'description'] = 'alp'
    labs.ix[(labs.description == 'alp'), 'unitofmeasure'] = 'u/l'

    return labs


def _pro_lymph(labs):
    """
    Clean Lymphocytes data 
    Removed 0.0 lymphocytes data row
    Normal values for the lymphocytes percentage is 28 to 55
    """
    labs = labs[~((labs.description == "lymphocytes") & (labs.clientresult == "0.0"))]
    labs = labs[~((labs.description == "lymphocytes") & (labs.clientresult == "0"))]

    return labs


def _pro_pot(labs):
    """
    potassium correction
    All units are in meq/l
    Normal range between 3.5-5.0 meq/l
    correct clientresults
    """
    pot = labs.description == 'potassium'
    labs.ix[pot, 'unitofmeasure'] = 'meq/l'
    labs = labs.drop(labs[(pot) & (labs.clientresult == 'to_follow')].index)
    labs.ix[labs.clientresult == '<20.0', 'clientresult'] = 20.0
    labs.ix[labs.clientresult == '>_10.0', 'clientresult'] = 10.0

    return labs

def _pro_bun(labs):
    """
    Correcting BUN(Blood Urea Nitrogen data)
    Normal range : 5 - 20 mg/dL
    BUN to creatinine ratio : 6-25
    On dialysis can have higher values as 40-60
    """
    labs.ix[((labs.description == "bun") & (labs.clientresult == "<_2")), 'clientresult'] = 2

    return labs


def _pro_anion_gap(labs):
    """
    anion gap correction
    All units are in meq/l
    Normal range between 3-11 meq/l
    serum anion gap range 8-16 meq/l
    <11 is generally considered normal
    urine anion gap >20 kidney unable to excrete ammonia
    if negative and the serum ag positive then gastro problems
    correct clientresults
    """
    ag = labs.description == 'anion_gap'
    labs.ix[ag, 'unitofmeasure'] = 'meq/l'
    labs.ix[(ag) & (labs.clientresult == '<5'), 'clientresult'] = 5.
    labs = fill_see_below(labs, ag)

    return labs


def pro_labs(dire=PROPATH, save=PROPATH):
    """
    Process train_RawVitalData.csv
    Basic processing
    Drop chstandard (62% null values)
    Drop clientresult canceled
    Remove descriptions
    Correct gfr columns
    Correct albumin columns
    Correct Allen columns
    Correct alp columns
    Correct pot columns
    Correct anion_gap columns
    """
    labs = pro_labs_basic(dire, None)

    del labs['chstandard']
    labs = labs[(labs.clientresult != 'canceled')]

    labs = _remove_desc(labs)

    labs = _pro_gfr(labs)
    labs = _pro_albumin(labs)
    labs = _pro_allen(labs)
    labs = _pro_alp(labs)
    labs = _pro_pot(labs)
    labs = _pro_anion_gap(labs)
    labs = _pro_lymph(labs)
    labs = _pro_bun(labs)

    if save:
        labs.to_csv(_get_path(save, 'labs.csv'), index=False)

    return labs


def process(dire=TRAINPATH, save=PROPATH):
    """
    Preprocesses all of the data
    """
    if not os.path.isdir(save):
        os.mkdir(save)
    icd_9_codes(dire, save)
    pro_label(dire, save)
    pro_static(dire, save)
    pro_vitals(dire, save)
    remove_rows(dire, save)
    correct_lab_data(save)
    pro_labs(save, save)


if __name__ == '__main__':
    process(*sys.argv[1:])
