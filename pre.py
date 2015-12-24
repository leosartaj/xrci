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


def bfill(labs, desc, key):
    """
    fills the 'key' clientresults
    uses bfill method
    bfill's only for same descriptions
    """
    see = labs.clientresult == key
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
    labs = labs[(labs.description != 'differential_information')]

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


def _pro_pot(labs):
    """
    potassium correction
    All units are in meq/l
    Normal range between 3.5-5.0 meq/l
    correct clientresults
    """
    pot = labs.description == 'potassium'
    labs.ix[(pot) & (labs.clientresult == 'to_follow'), 'clientresult'] = np.nan
    return labs


def _pro_clean_clientresults(labs):
    """
    creatine_kinase

    Corrected Absolute_neutrophill_count_automated
    normal range : 1.5 to 8.00

    urobilinogen correction
    range 0-8 mg/dl (units given are different but range appears same)

    specific_gravity correction
    density of urine to water
    range 1.002-1.030 if kidneys normal

    Correct Phosphorus clientresult
    normal range : 2.4 - 4.1 mg/dL

    Correct magnesium clientresult
    Normal range : 1.5 - 2.5

    Correct protein clientresult
    normal range : 6.0 to 8.3 mg/dL

    Globulin ratio -> globulin to albumin ratio (range 1:2, 1.7-2.2 also ok)
    Globulin correction
    unitofmeasure g/dl
    Lot of ranges provided, 2.3-3.5 g/dl

    Corrected clientresult in bilirubin
    Normal range : 0.3 to 1.9 mg/dL

    Co2 content correction
    Normal range between 23-30 meq/l

    Chloride clientresult and unitofmeasure correction
    Normal range : 96 - 106 mEq/l
    Metabolic acidosis

    Correcting BUN(Blood Urea Nitrogen data)
    Normal range : 5 - 20 mg/dL
    BUN to creatinine ratio : 6-25
    On dialysis can have higher values as 40-60

    Clean Lymphocytes data
    Removed 0.0 lymphocytes data row
    Normal values for the lymphocytes percentage is 28 to 55

    albumin correction
    Range 3.5-5.5 g/dL (35-55 g/L) is normal

    Very related tests
    alt (sgpt) correction
    Normal range between 10-40 u/l for males and 7-35 u/l for females
    ast (sgot) correction
    Normal range between 14-20 u/l for males and 10-36 u/l for females
    correct clientresults

    anion gap correction
    Normal range between 3-11 meq/l
    serum anion gap range 8-16 meq/l
    <11 is generally considered normal
    urine anion gap >20 kidney unable to excrete ammonia
    if negative and the serum ag positive then gastro problems

    Creatinine clientresult corrected
    Normal values : 0.51 - 1.2
    
    Corrected mch
    normal range : 27-36

	Corrected Eosinophils
    normal range : 0 to 6.0 %

    Corrected mcv data
    normal range : 77 - 95
    
    Corrected hematocrit
    normal range : 38.8 - 50 %

    Corrected hgb
    normal range :  12.0 to 17.5

    Monocytes
    normal range : 0.0 - 13.0

    mpv
    normal range : 7.5 - 11.5

    platelet_count
    normal range : 150 - 450

    rbc
    normal range : 4.2 - 5.4 million/ul 
    
    rdw
    normal range : 11.5 - 14.5 %
    """

    labs.ix[labs.clientresult == "----", 'clientresult'] = np.nan

    glo = labs.description == 'globulin'
    labs.ix[(glo) & (labs.clientresult == '-2.2'), 'clientresult'] = 2.2

    labs.ix[(labs.clientresult == '4.0_g/dl'), 'clientresult'] = 4.0
    labs.ix[(labs.clientresult == '>_3.2_-_normal'), 'clientresult'] = 4.0
    labs.ix[(labs.clientresult == '<1.5'), 'clientresult'] = 1.5
    labs.ix[(labs.clientresult == '---__11/25/11_0858_---_mch_previously_reported_as:___21.1__l_pg'), 'clientresult'] = 21.1
    labs.ix[(labs.clientresult == '---__11/25/11_0858_---_mcv_previously_reported_as:___70.0__l_fl'), 'clientresult'] = 70.0
    labs.ix[(labs.clientresult == '---__11/25/11_0858_---_mpv_previously_reported_as:___10.4_fm'), 'clientresult'] = 10.4
    labs.ix[(labs.clientresult == '---__11/25/11_0858_---_rdw_previously_reported_as:___17.6__h_%'), 'clientresult'] = 17.6

    labs = bfill(labs, 'co2_content', 'see_below')
    labs = bfill(labs, 'alt_(sgpt)', 'see_below')
    labs = bfill(labs, 'ast_(sgot)', 'see_below')
    labs = bfill(labs, 'anion_gap', 'see_below')
    labs = bfill(labs, 'creatinine_(enz)', 'see_below')
    labs = bfill(labs, 'mch', 'see_below')
    labs = bfill(labs, 'hct', 'see_below')
    labs = bfill(labs, 'hgb', 'see_below')
    labs = bfill(labs, 'mpv', 'see_below')
    labs = bfill(labs, 'platelet_count', 'see_below')
    labs = bfill(labs, 'rbc', 'see_below')

    lym = labs.description == 'lymphocytes'
    labs.ix[(lym) & (labs.clientresult == "0.0"), 'clientresult'] = np.nan
    labs.ix[(lym) & (labs.clientresult == "0"), 'clientresult'] = np.nan
    labs.ix[labs.clientresult == 'specimen_drawn_from_arterial_line.', 'clientresult'] = np.nan

    rbc = labs.description == 'rbc'
    labs.ix[(rbc) & (labs.clientresult == 'none_seen'), 'clientresult'] = np.nan
    labs.ix[(rbc) & (labs.clientresult == 'none_seen'), 'clientresult'] = np.nan
    ch = ['<', '>', '_', '=']
    for c in ch:
        labs.ix[labs.clientresult.str[0] == c, 'clientresult'] = labs.ix[labs.clientresult.str[0] == c, 'clientresult'].str[1:]

    return labs


def _pro_cat(labs):
    """
    Label categorical data with codes
    hemolysis_index, icteric_index, lipemia_index correction

    amorphous_urates correction
    crystals in urine, higher is not good

    bacteria correction
    epithelial_cells correction

    sample type labs
    no unit of measure
    replaced other clientresult values with NaN
    mapped remaining values to integers

    sample site labs
    no unit of measure
    replaced unknown and other clientresult values with NaN
    mapped remaining values to integers

    protein_qualitative correction
    lower value is normal, higher is troublesome for the patient

    Corrected glucose values
    Range : 70 -100 mg/dL

    Allen's Test Correction
    Results are Pass(1), Fail(0) or Half Passed 0.5 (fail = 0,else =1)
    """
    hemo = labs.description == 'hemolysis_index'
    labs.ix[(hemo) & (labs.clientresult == 'no_hemolysis'), 'clientresult'] = 0
    labs.ix[(hemo) & (labs.clientresult == 'slightly'), 'clientresult'] = 1
    labs.ix[(hemo) & (labs.clientresult == 'moderately'), 'clientresult'] = 2
    labs.ix[(hemo) & (labs.clientresult == 'grossly'), 'clientresult'] = 3
    labs.ix[(hemo) & (labs.clientresult == 'highly'), 'clientresult'] = 4

    ice = labs.description == 'icteric_index'
    labs.ix[(ice) & (labs.clientresult == 'not_icteric'), 'clientresult'] = 0
    labs.ix[(ice) & (labs.clientresult == 'slightly'), 'clientresult'] = 1
    labs.ix[(ice) & (labs.clientresult == 'moderately'), 'clientresult'] = 2
    labs.ix[(ice) & (labs.clientresult == 'grossly'), 'clientresult'] = 3
    labs.ix[(ice) & (labs.clientresult == 'highly'), 'clientresult'] = 4

    lip = labs.description == 'lipemia_index'
    labs.ix[(lip) & (labs.clientresult == 'no_lipemia'), 'clientresult'] = 0
    labs.ix[(lip) & (labs.clientresult == 'slightly'), 'clientresult'] = 1
    labs.ix[(lip) & (labs.clientresult == 'moderately'), 'clientresult'] = 2
    labs.ix[(lip) & (labs.clientresult == 'grossly'), 'clientresult'] = 3
    labs.ix[(lip) & (labs.clientresult == 'highly'), 'clientresult'] = 4

    au = labs.description == 'amorphous_urates'
    labs.ix[(au) & (labs.clientresult == 'none_seen'), 'clientresult'] = 0
    labs.ix[(au) & (labs.clientresult == 'rare'), 'clientresult'] = 1
    labs.ix[(au) & (labs.clientresult == 'few'), 'clientresult'] = 2
    labs.ix[(au) & (labs.clientresult == 'moderate'), 'clientresult'] = 3
    labs.ix[(au) & (labs.clientresult == 'many'), 'clientresult'] = 4
    labs.ix[(au) & (labs.clientresult == 'massive'), 'clientresult'] = 5

    bac = labs.description == 'bacteria'
    labs.ix[(bac) & (labs.clientresult == 'none_seen'), 'clientresult'] = 0
    labs.ix[(bac) & (labs.clientresult == 'rare'), 'clientresult'] = 1
    labs.ix[(bac) & (labs.clientresult == 'few'), 'clientresult'] = 2
    labs.ix[(bac) & (labs.clientresult == 'moderate'), 'clientresult'] = 3
    labs.ix[(bac) & (labs.clientresult == 'many'), 'clientresult'] = 4
    labs.ix[(bac) & (labs.clientresult == 'massive'), 'clientresult'] = 5

    ec = labs.description == 'epithelial_cells'
    labs.ix[(ec) & (labs.clientresult == 'none_seen'), 'clientresult'] = 0
    labs.ix[(ec) & (labs.clientresult == 'rare'), 'clientresult'] = 1
    labs.ix[(ec) & (labs.clientresult == 'occasional'), 'clientresult'] = 2
    labs.ix[(ec) & (labs.clientresult == 'few'), 'clientresult'] = 3
    labs.ix[(ec) & (labs.clientresult == 'moderate'), 'clientresult'] = 4
    labs.ix[(ec) & (labs.clientresult == 'many'), 'clientresult'] = 5
    labs.ix[(ec) & (labs.clientresult == 'massive'), 'clientresult'] = 6

    samp = labs.description == 'sampletype'
    labs.ix[samp & (labs.clientresult == 'other'), 'clientresult'] = np.nan
    labs.ix[samp & (labs.clientresult == 'arterial'), 'clientresult'] = 0
    labs.ix[samp & (labs.clientresult == 'venous'), 'clientresult'] = 1
    labs.ix[samp & (labs.clientresult == 'mixed_venous'), 'clientresult'] = 2
    labs.ix[samp & (labs.clientresult == 'cord_ven'), 'clientresult'] = 3
    labs.ix[samp & (labs.clientresult == 'cord_art'), 'clientresult'] = 4

    samp = labs.description == 'samplesite'
    labs.ix[samp & (labs.clientresult == 'unknown'), 'clientresult'] = np.nan
    labs.ix[samp & (labs.clientresult == 'other'), 'clientresult'] = np.nan
    labs.ix[samp & (labs.clientresult == 'r_radial'), 'clientresult'] = 0
    labs.ix[samp & (labs.clientresult == 'a-line'), 'clientresult'] = 1
    labs.ix[samp & (labs.clientresult == 'l_radial'), 'clientresult'] = 2
    labs.ix[samp & (labs.clientresult == 'r_brachial'), 'clientresult'] = 3
    labs.ix[samp & (labs.clientresult == 'swan'), 'clientresult'] = 4
    labs.ix[samp & (labs.clientresult == 'r_femoral'), 'clientresult'] = 5
    labs.ix[samp & (labs.clientresult == 'ua-line'), 'clientresult'] = 6
    labs.ix[samp & (labs.clientresult == 'cord'), 'clientresult'] = 7
    labs.ix[samp & (labs.clientresult == 'l_brachial'), 'clientresult'] = 8
    labs.ix[samp & (labs.clientresult == 'l_femoral'), 'clientresult'] = 9

    pq = labs.description == 'protein_qualitative'
    labs.ix[(pq) & (labs.clientresult == 'negative'), 'clientresult'] = 0
    labs.ix[(pq) & (labs.clientresult == 'neg'), 'clientresult'] = 0
    labs.ix[(pq) & (labs.clientresult == 'trace'), 'clientresult'] = 0.5
    labs.ix[(pq) & (labs.clientresult == '1+'), 'clientresult'] = 1
    labs.ix[(pq) & (labs.clientresult == '2+'), 'clientresult'] = 2
    labs.ix[(pq) & (labs.clientresult == '3+'), 'clientresult'] = 3

    glu = labs.description == 'glucose'
    labs.ix[((glu) & (labs.clientresult == 'slight_hemolysis')), 'clientresult'] = np.nan
    labs.ix[(glu) & (labs.clientresult == '2+'), 'clientresult'] = 200
    labs.ix[(glu) & (labs.clientresult == 'negative'), 'clientresult'] = 85
    labs.ix[(glu) & (labs.clientresult == '1+'), 'clientresult'] = 100
    labs.ix[(glu) & (labs.clientresult == '3+'), 'clientresult'] = 300
    labs.ix[(glu) & (labs.clientresult == 'trace'), 'clientresult'] = 100
    labs.ix[(glu) & (labs.clientresult == 'neg'), 'clientresult'] = 85

    labs.ix[((labs.description == "allen's_test") & (labs.clientresult == "collected_by_respiratory.")), 'clientresult'] = 1
    labs.ix[((labs.description == "allen's_test") & (labs.clientresult.isnull())), 'clientresult'] = 1
    labs.ix[((labs.description == "allen's_test") & (labs.clientresult == "na")), 'clientresult'] = 1
    labs.ix[((labs.description == "allen's_test") & (labs.clientresult == ".")), 'clientresult'] = 1
    labs.ix[((labs.description == "allen's_test") & (labs.clientresult == "passed_left_radial")), 'clientresult'] = 0.5
    labs.ix[((labs.description == "allen's_test") & (labs.clientresult == "pass")), 'clientresult'] = 1
    labs.ix[((labs.description == "allen's_test") & (labs.clientresult == "fail")), 'clientresult'] = 0

    return labs


def pro_labs(dire=PROPATH, save=PROPATH):
    """
    Process train_RawVitalData.csv
    Basic processing
    Drop chstandard (62% null values)
    Drop clientresult canceled
    Remove descriptions
    Correct gfr columns
    Correct pot columns
    Correct categorical columns
    Correct clientresults
    """
    labs = pro_labs_basic(dire, None)

    del labs['chstandard']
    labs = labs[(labs.clientresult != 'canceled')]

    labs = _remove_desc(labs)

    labs = _pro_gfr(labs)
    labs = _pro_pot(labs)
    labs = _pro_cat(labs)
    labs = _pro_clean_clientresults(labs)

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
