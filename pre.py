"""
Preprocessing of data
"""

import sys
import numpy as np
import pandas as pd

from util import get_path, TRAINPATH, PROPATH


def icd_9_codes(dire=TRAINPATH, save=PROPATH):
    """
    Creates a dataframe mapping icd_9 codes
    to names and descriptions

    returns: pd.DataFrame
    """
    labels = pd.read_csv(get_path(dire, 'train_label.csv'))
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
        data.to_csv(get_path(save, 'icd_9.csv'), index=False)

    return data


def pro_label(dire=TRAINPATH, save=PROPATH):
    """
    Process train_labels.csv
    Sort by Id
    Replace icd_9 codes with the appropriate names
    Replace NaN instances with -1.
    """
    labels = pd.read_csv(get_path(dire, 'train_label.csv'))

    lsort = labels.sort_values('id_', ascending=True)
    lsort = lsort.reset_index()
    del lsort['index']

    icd_9 = pd.read_csv(get_path(save, 'icd_9.csv'))
    assert all(lsort.columns[1:]) == all(icd_9.icd_9)
    lsort.columns = np.concatenate([['id'], icd_9.names])
    assert len(lsort.columns) == 20

    lsort = lsort.fillna(-1)
    if save:
        lsort.to_csv(get_path(save, 'label.csv'), index=False)

    return lsort


def pro_static(dire=TRAINPATH, save=PROPATH):
    """
    Process train_Static_data.csv
    Sort by Id
    lower all the column names
    lower all the column values of gender, maritalstatus, ethnicgroup,
    admitspeciality
    """
    static = pd.read_csv(get_path(dire, 'train_Static_data.csv'))

    static = static.sort_values('id', ascending=True)
    static = static.reset_index()
    del static['index']

    static.columns = [col.lower() for col in static.columns]
    static = static.applymap(lambda x: x.lower().replace(' ', '_').
                             replace('-', '_') if isinstance(x, str) else x)

    if save:
        static.to_csv(get_path(save, 'static.csv'), index=False)

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
    vitals = pd.read_csv(get_path(dire, 'train_RawVitalData.csv'))

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
        vitals.to_csv(get_path(save, 'vitals.csv'), index=False)

    return vitals


def remove_rows(dire=TRAINPATH, save=PROPATH):
    """
    Returns the cleaned labs df and dirty rows in train_RawLabData.csv
    Computes pd.DataFrame again and again until error free.
    Very poor efficiency. Avoid it's usage.
    """
    fpath = get_path(dire, 'train_RawLabData.csv')
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
        labs.to_csv(get_path(save, 'labs_cut.csv'), index=False)

    return labs, rows


def correct_lab_data(dire=PROPATH):
    """
    Removes Irregularities in CSV train_RawLabData.csv
    call remove_rows first
    """
    files = [get_path(dire,'labs_cut.csv'), 'corrected_lab_data.csv']
    with open(get_path(dire,'labs_correct.csv'),'w') as newFile:
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
    Drop duplicates
    """
    labs = pd.read_csv(get_path(dire, 'labs_correct.csv'))

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

    labs = labs.drop_duplicates()

    if save:
        labs.to_csv(get_path(save, 'labs_basic.csv'), index=False)

    return labs


def remove_desc(dire=PROPATH, save=PROPATH):
    labs = pd.read_csv(get_path(dire, 'labs_basic.csv'))

    desc = []
    with open('remove_labs.txt') as f:
        for line in f.readlines():

            if not line.startswith('#') and line != '\n':
                if line.endswith('\n'):
                    line = line[:-1]
                desc.append(line)

    rem = labs.description == desc[0]
    for d in desc[1:]:
        rem = rem | (labs.description == d)

    labs = labs[~rem]

    if save:
        labs.to_csv(get_path(save, 'labs_remove.csv'), index=False)

    return labs


def _pro_gfr(labs):
    """
    gfr correction
    estimated_gfr-african_american and estimated_gfr-other are very similar
    egfr_non-african_american and egfr_african_american are very similar
    rename the other two as gfr
    clean up values in clientresult
    """
    gfr = labs[labs.clientresult == '>60'].description.unique()

    labs = labs[labs.description != gfr[1]]

    labs = labs[labs.description != gfr[3]]

    labs.ix[(labs.description == gfr[0]) | (labs.description == gfr[2]),
            'description'] = 'gfr'

    labs.ix[(labs.description == 'gfr') &
            ((labs.clientresult == '>60') |
             (labs.clientresult == '>60.00')), 'clientresult'] = 75

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

    aptt
    normal range : 70 - 120 secs (liver diseases)

    inr
    normal range : 0.8 -2.0 (heart related ) those who have mechanical heart can have 2-3

    prothrombin_time
    normal range : 12-13 secs

    sedimentation_rate
    normal range : 0 - 29 mm/hr

    potassium correction
    All units are in meq/l
    Normal range between 3.5-5.0 meq/l
    correct clientresults
    """

    labs.ix[labs.clientresult == "----", 'clientresult'] = np.nan
    labs.ix[(labs.clientresult == '&#x20;'), 'clientresult'] = np.nan
    labs.ix[(labs.clientresult == 'see_below'), 'clientresult'] = np.nan
    labs.ix[(labs.clientresult == 'to_follow'), 'clientresult'] = np.nan
    labs.ix[(labs.clientresult == '*'), 'clientresult'] = np.nan

    glo = labs.description == 'globulin'
    labs.ix[(glo) & (labs.clientresult == '-2.2'), 'clientresult'] = 2.2

    labs.ix[(labs.clientresult == '4.0_g/dl'), 'clientresult'] = 4.0
    labs.ix[(labs.clientresult == '>_3.2_-_normal'), 'clientresult'] = 4.0
    labs.ix[(labs.clientresult == '<1.5'), 'clientresult'] = 1.5
    labs.ix[(labs.clientresult == '---__11/25/11_0858_---_mch_previously_reported_as:___21.1__l_pg'), 'clientresult'] = 21.1
    labs.ix[(labs.clientresult == '---__11/25/11_0858_---_mcv_previously_reported_as:___70.0__l_fl'), 'clientresult'] = 70.0
    labs.ix[(labs.clientresult == '---__11/25/11_0858_---_mpv_previously_reported_as:___10.4_fm'), 'clientresult'] = 10.4
    labs.ix[(labs.clientresult == '---__11/25/11_0858_---_rdw_previously_reported_as:___17.6__h_%'), 'clientresult'] = 17.6
    labs.ix[labs.clientresult == 'cannot_perform_cell_count_due_to_degeneration_of_cells.""', 'clientresult'] = np.nan
    labs.ix[labs.clientresult == 'unable_to_determine_differential_due_to_distortion_of_white_blood_cells', 'clientresult'] = np.nan
    labs.ix[labs.clientresult == 'unable_to_preform_axccurate_test_because_of_mucoid_specimen._few_rbc_observed_on_wet_prep.', 'clientresult'] = np.nan
    labs.ix[labs.clientresult == 'uanable_to_count_because_of_mucoid_consistency.__wet_prep_show_massive_clumps_of_wbc,_few_rbc_observed.__many_bacteria_see.', 'clientresult'] = np.nan
    labs.ix[labs.clientresult == 'cannot_perform_cell_count_due_to_degeneration_of_cells.""', 'clientresult'] = np.nan
    labs.ix[labs.clientresult == 'unable_to_determine_differential_due_to_distortion_of_white_blood_cells', 'clientresult'] = np.nan
    labs.ix[labs.clientresult == 'unable_to_preform_axccurate_test_because_of_mucoid_specimen._few_rbc_observed_on_wet_prep.', 'clientresult'] = np.nan
    labs.ix[labs.clientresult == 'uanable_to_count_because_of_mucoid_consistency.__wet_prep_show_massive_clumps_of_wbc,_few_rbc_observed.__many_bacteria_see.', 'clientresult'] = np.nan
    labs.ix[labs.clientresult == "date_/_time_next_dose_:_unknown", 'clientresult'] = np.nan
    labs.ix[labs.clientresult == "lc_results_scanned_in", 'clientresult'] = np.nan
    labs.ix[labs.clientresult == "---__11/25/11_0858_---_plat_previously_reported_as:___116__l_mm3", 'clientresult'] = np.nan
    labs.ix[labs.clientresult == "---__12/18/11_0745_---_semi-quant_gluc_previously_reported_as:___51__l", 'clientresult'] = np.nan
    labs.ix[labs.clientresult == "trough_vanc_date_/_time_next_dose_:_unknown", 'clientresult'] = np.nan
    labs.ix[labs.clientresult == "---__12/22/11_0656_---_ly#m_previously_reported_as:___1.2_mm3", 'clientresult'] = np.nan
    labs.ix[labs.clientresult == "---__12/22/11_0656_---_ly%m_previously_reported_as:___10_#_l_%", 'clientresult'] = np.nan
    labs.ix[labs.clientresult == "---__12/22/11_0657_---_mo#m_previously_reported_as:___1.2__h_mm3", 'clientresult'] = np.nan
    labs.ix[labs.clientresult == "---__12/22/11_0656_---_mo%m_previously_reported_as:___10_#_%", 'clientresult'] = np.nan
    labs.ix[labs.clientresult == "---__12/22/11_0656_---_ne#m_previously_reported_as:___9.4__h_mm3", 'clientresult'] = np.nan
    labs.ix[labs.clientresult == "---__12/22/11_0656_---_ne%m_previously_reported_as:___79_#_h_%", 'clientresult'] = np.nan
    labs.ix[labs.clientresult == "---__12/22/11_0657_---_eos#m_previously_reported_as:___0.1_mm3", 'clientresult'] = np.nan
    labs.ix[labs.clientresult == "---__12/22/11_0656_---_eos%m_previously_reported_as:___1_#_%", 'clientresult'] = np.nan
    labs.ix[(labs.clientresult == "see_scanned_report_in_emr"), 'clientresult'] = np.nan
    labs.ix[(labs.clientresult == '2-5'), 'clientresult'] = 3.5
    labs.ix[(labs.clientresult == '1_/hpf'), 'clientresult'] = 1

    lym = labs.description == 'lymphocytes'
    labs.ix[(lym) & (labs.clientresult == "0.0"), 'clientresult'] = np.nan
    labs.ix[(lym) & (labs.clientresult == "0"), 'clientresult'] = np.nan
    labs.ix[labs.clientresult == 'specimen_drawn_from_arterial_line.', 'clientresult'] = np.nan

    rbc = labs.description == 'rbc'
    labs.ix[(rbc) & (labs.clientresult == 'none_seen'), 'clientresult'] = np.nan
    labs.ix[(rbc) & (labs.clientresult == 'none_seen'), 'clientresult'] = np.nan

    labs.ix[(labs.clientresult == 'unpt'), 'clientresult'] = np.nan

    ms = labs.description == 'm-spike'
    labs.ix[((ms) & (labs.clientresult == "comment:")), 'clientresult'] = np.nan
    labs.ix[((ms) & (labs.clientresult == "not_observed")), 'clientresult'] = 0

    cdt = labs.description == 'm-spike,%'
    labs.ix[((cdt) & (labs.clientresult == "not_observed")), 'clientresult'] = np.nan

    ri = labs.description == 'rifampin'
    labs.ix[((ri) & (labs.clientresult == "1_<=1")), 'clientresult'] = np.nan

    ch = ['<', '>', '_', '=', '_']
    for c in ch:
        labs.ix[labs.clientresult.str[0] == c, 'clientresult'] = labs.ix[labs.clientresult.str[0] == c, 'clientresult'].str[1:]

    ch = ['g/dl', '_', '_m', '+']
    for c in ch:
        labs.ix[labs.clientresult.str[-len(c):] == c, 'clientresult'] = labs.ix[labs.clientresult.str[-len(c):] == c, 'clientresult'].str[:-len(c)]

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

    Indexed anisocytosis values

    Indexed microcytic values

    Indexed ovalocytes values

    Indexed poikilocytosis values

    Indexed polychromasia values

    mapped c._difficile_dna_pcr values

    hiv_ab/ag correction
    hepatitis_b_sur_ab correction
    hepatitis_b_core_antibody correction

    abo_intep correction
    quantiferon_tb_gold correction

    Ketones urine
    Results are negative(0), trace(1), 1+(2), 2+(3), 3+(4)

    leukocytes_esterase
    Results are negative(0), trace(1), 1+(2), 2+(3)

    nitrites
    Resuls are negative(0), positive(1)

    bacteria,_auto
    mapped none 0, innumerable 10

    casts,_manual
    mapped strings

    mapped strings
    positive_screen 1
    negative_screen 0

    hyaline_casts,_manual
    mapped innumerable 200

    occult_blood,_fecal_#1
    cleaned

    c.difficile_toxin
    cleaned

    hbsag_screen
    cleaned

    hcv_ab
    cleaned

    hep_b_core_ab,_igm
    cleaned

    poc_urine_pregnancy_result
    cleaned

    hepatitis_a_ab,_total
    cleaned

    acetones_/_ketones
    cleaned
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

    ket = labs.description == 'ketones,_urine'
    labs.ix[ket & (labs.clientresult == 'negative'), 'clientresult'] = 0
    labs.ix[ket & (labs.clientresult == 'neg'), 'clientresult'] = 0
    labs.ix[ket & (labs.clientresult == 'trace'), 'clientresult'] = 1
    labs.ix[ket & (labs.clientresult == '1+'), 'clientresult'] = 2
    labs.ix[ket & (labs.clientresult == '2+'), 'clientresult'] = 3
    labs.ix[ket & (labs.clientresult == '3+'), 'clientresult'] = 4

    leu = labs.description == 'leukocytes_esterase'
    labs.ix[leu & (labs.clientresult == 'negative'), 'clientresult'] = 0
    labs.ix[leu & (labs.clientresult == 'neg'), 'clientresult'] = 0
    labs.ix[leu & (labs.clientresult == 'trace'), 'clientresult'] = 1
    labs.ix[leu & (labs.clientresult == '1+'), 'clientresult'] = 2
    labs.ix[leu & (labs.clientresult == '2+'), 'clientresult'] = 3

    nit = ((labs.description == 'nitrites') | (labs.description == "urn/csf_streptococcal_antigen") | (labs.description == 'c._difficile_dna_pcr')
            | (labs.description == 'hiv_ag/ab') | (labs.description == 'stool_occult_blood_1') | (labs.description == 'occult_blood,_fecal_#1')
            | (labs.description == 'hep_b_core_ab,_igm') | (labs.description == 'poc_urine_pregnancy_result')
            | (labs.description == 'hepatitis_a_ab,_total') | (labs.description == 'antinuclear_antibodies')
            | (labs.description == 'c.difficile_toxin' )| (labs.description == 'blastomyces_dermat._cf')
            | (labs.description == 'poc_nitrazine') | (labs.description == 'e.chaffeensis_igm_titer')
            | (labs.description == 'poc_occult_blood_result') | (labs.description == 'occult_blood,_gastric')
            | (labs.description == 'poc_strep,_quick_result')| (labs.description == 'rotavirus,_stool')
            | (labs.description == 'urn/csf_strep_pneumo_antigen') | (labs.description == 'benzodiazepines')
            | (labs.description == 'c.difficile_toxins_a_b,_eia') | (labs.description == 'direct_strep_a,_culture_if_neg')
            | (labs.description == 'poc_nitrazine') | (labs.description == 'e.chaffeensis_igm_titer')
            | (labs.description == 'poc_strep,_quick_result')
            | (labs.description == 'rotavirus,_stool') | (labs.description == 'urn/csf_strep_pneumo_antigen') | (labs.description == 'benzodiazepines')
            | (labs.description == 'c.difficile_toxins_a_b,_eia') | (labs.description == 'direct_strep_a,_culture_if_neg')
            | (labs.description == 'urine_benzodiazepine') | (labs.description == 'rh') | (labs.description == 'smooth_muscle_ab'))

    labs.ix[nit & (labs.clientresult == 'negative'), 'clientresult'] = 0
    labs.ix[nit & (labs.clientresult == 'tnp'), 'clientresult'] = np.nan
    labs.ix[nit & (labs.clientresult == 'neg'), 'clientresult'] = 0
    labs.ix[nit & (labs.clientresult == 'equivocal'), 'clientresult'] = 0.5
    labs.ix[nit & (labs.clientresult == 'positive'), 'clientresult'] = 1
    labs.ix[nit & (labs.clientresult == 'pos'), 'clientresult'] = 1

    mic = ((labs.description == 'anisocytosis') | (labs.description == 'microcytic') | (labs.description == 'ovalocytes') |
            (labs.description == 'poikilocytosis') | (labs.description == 'polychromasia') | (labs.description == 'macrocytosis')
            | (labs.description == 'toxic_vacuolation') | (labs.description == 'burr_cells') | (labs.description == 'schistocytes')
            | (labs.description == 'hypochromia') | (labs.description == 'target_cells') | (labs.description == 'basophilic_stippling')
            | (labs.description == 'microcytosis') | (labs.description == 'legionella_pneu_urinary_ag') | (labs.description == 'urine_sperm')
            | (labs.description == 'yeast') | (labs.description == 'dohle_bodies'))

    labs.ix[((mic) & (labs.clientresult == 'rare')), 'clientresult'] = 0.5
    labs.ix[((mic) & (labs.clientresult == 'few')), 'clientresult'] = 1
    labs.ix[((mic) & (labs.clientresult == 'slight')), 'clientresult'] = 1
    labs.ix[((mic) & (labs.clientresult == 'slight-mod')), 'clientresult'] = 1.5
    labs.ix[((mic) & (labs.clientresult == 'mild')), 'clientresult'] = 1.5
    labs.ix[((mic) & (labs.clientresult == 'moderate')), 'clientresult'] = 2
    labs.ix[((mic) & (labs.clientresult == 'many')), 'clientresult'] = 2.5
    labs.ix[((mic) & (labs.clientresult == 'marked')), 'clientresult'] = 3

    hbab = (labs.description == 'hepatitis_b_sur_ab')
    labs.ix[((hbab) & (labs.clientresult == "non-reactive")), 'clientresult'] = 0
    labs.ix[((hbab) & (labs.clientresult == "reactive")), 'clientresult'] = 1

    ha = labs.description == 'hepatitis_b_core_antibody'
    labs.ix[((ha) & (labs.clientresult == "grayzone")), 'clientresult'] = np.nan
    labs.ix[((ha) & ((labs.clientresult == "non_reactive") | (labs.clientresult == "nonreactive"))), 'clientresult'] = 0
    labs.ix[((ha) & (labs.clientresult == "reactive")), 'clientresult'] = 1

    abo = labs.description == 'abo_intep'
    labs.ix[((abo) & (labs.clientresult == "o")), 'clientresult'] = 0
    labs.ix[((abo) & (labs.clientresult == "a")), 'clientresult'] = 1
    labs.ix[((abo) & (labs.clientresult == "b")), 'clientresult'] = 2
    labs.ix[((abo) & (labs.clientresult == "ab")), 'clientresult'] = 3

    qtg = labs.description == 'quantiferon_tb_gold'
    labs.ix[((qtg) & (labs.clientresult == "indeterminate")), 'clientresult'] = np.nan
    labs.ix[((qtg) & (labs.clientresult == "negative")), 'clientresult'] = 0
    labs.ix[((qtg) & (labs.clientresult == "positive")), 'clientresult'] = 1

    sta = labs.description == 'stool_appearance'
    labs.ix[((sta) & (labs.clientresult == "liquid")), 'clientresult'] = 0
    labs.ix[((sta) & (labs.clientresult == "loose")), 'clientresult'] = 0.5
    labs.ix[((sta) & (labs.clientresult == "greasy")), 'clientresult'] = 1
    labs.ix[((sta) & (labs.clientresult == "semi-liquid")), 'clientresult'] = 1.5
    labs.ix[((sta) & (labs.clientresult == "soft")), 'clientresult'] = 2
    labs.ix[((sta) & (labs.clientresult == "formed_stool")), 'clientresult'] = 2.5
    labs.ix[((sta) & (labs.clientresult == "solid")), 'clientresult'] = 3
    labs.ix[((sta) & (labs.clientresult == "firm")), 'clientresult'] = 3.5
    labs.ix[((sta) & (labs.clientresult == "rcvd_on_card")), 'clientresult'] = 4

    stc = labs.description == 'stool_color'
    labs.ix[((stc) & (labs.clientresult == "greenish_brown")), 'clientresult'] = 0
    labs.ix[((stc) & (labs.clientresult == "dark_brown")), 'clientresult'] = 1
    labs.ix[((stc) & (labs.clientresult == "light_brown")), 'clientresult'] = 2
    labs.ix[((stc) & (labs.clientresult == "brown")), 'clientresult'] = 3
    labs.ix[((stc) & (labs.clientresult == "scant_sample")), 'clientresult'] = 4
    labs.ix[((stc) & (labs.clientresult == "green")), 'clientresult'] = 5

    stb = labs.description == 'stool_occult_blood_1'
    labs.ix[((stb) & (labs.clientresult == "negative")), 'clientresult'] = 0
    labs.ix[((stb) & (labs.clientresult == "positive")), 'clientresult'] = 1

    bct = labs.description == 'bacteria,_auto'
    labs.ix[((bct) & (labs.clientresult == "none")), 'clientresult'] = 0
    labs.ix[((bct) & (labs.clientresult == "innumerable")), 'clientresult'] = 1

    cm = labs.description == 'casts,_manual'
    labs.ix[((cm) & (labs.clientresult == "hyaline_casts")), 'clientresult'] = 0
    labs.ix[((cm) & (labs.clientresult == "coarsely_gran._cast")), 'clientresult'] = 1
    labs.ix[((cm) & (labs.clientresult == "finely_granular_cast")), 'clientresult'] = 2
    labs.ix[((cm) & (labs.clientresult == "white_cell_casts")), 'clientresult'] = 3
    labs.ix[((cm) & (labs.clientresult == "waxy_casts")), 'clientresult'] = 4

    labs.ix[labs.clientresult == "positive_screen", 'clientresult'] = 1
    labs.ix[labs.clientresult == "negative_screen", 'clientresult'] = 0

    hcm = labs.description == 'hyaline_casts,_manual'
    labs.ix[((hcm) & (labs.clientresult == "innumerable")), 'clientresult'] = 200

    fsw = labs.description == 'fecal_smear_for_wbc'
    labs.ix[((fsw) & (labs.clientresult == "fecal_wbc:_no_white_blood_cells")), 'clientresult'] = 0
    labs.ix[((fsw) & (labs.clientresult == "fecal_wbc:_rare_white_blood_cells")), 'clientresult'] = 1
    labs.ix[((fsw) & (labs.clientresult == "fecal_wbc:_few_white_blood_cells")), 'clientresult'] = 2
    labs.ix[((fsw) & (labs.clientresult == "fecal_wbc:_many_white_blood_cells")), 'clientresult'] = 3

    hab = labs.description == 'hcv_ab'
    labs.ix[((hab) & (labs.clientresult == "see_scanned_report_in_emr"))] = np.nan

    cdt = labs.description == 'm-spike,%'
    labs.ix[((cdt) & (labs.clientresult == "not_observed")), 'clientresult'] = np.nan

    ak = labs.description == "acetones_/_ketones"
    labs.ix[((ak) & (labs.clientresult == "negative")), 'clientresult'] = 0
    labs.ix[((ak) & (labs.clientresult == "pos_1:8_small")), 'clientresult'] = 1
    labs.ix[((ak) & (labs.clientresult == "pos_1:16_moderate")), 'clientresult'] = 2
    labs.ix[((ak) & (labs.clientresult == "pos_1:16_large")), 'clientresult'] = 3

    csa = labs.description == "csf_appearance"
    labs.ix[((csa) & (labs.clientresult == "clear")), 'clientresult'] = 0
    labs.ix[((csa) & (labs.clientresult == "hazy")), 'clientresult'] = 1

    csc = labs.description == "csf_color"
    labs.ix[((csc) & (labs.clientresult == "colorless")), 'clientresult'] = 0
    labs.ix[((csc) & (labs.clientresult == "yellow")), 'clientresult'] = 1
    labs.ix[((csc) & (labs.clientresult == "pink")), 'clientresult'] = 2

    cdt = labs.description == 'heparinized_sample'
    labs.ix[((cdt) & (labs.clientresult == "yes")), 'clientresult'] = 1
    labs.ix[((cdt) & (labs.clientresult == "no")), 'clientresult'] = 0

    ctn = labs.description == "csf_tube_number"
    labs.ix[((ctn) & (labs.clientresult == "tube_#1")), 'clientresult'] = 1
    labs.ix[((ctn) & (labs.clientresult == "tube_#2")), 'clientresult'] = 2
    labs.ix[((ctn) & (labs.clientresult == "tube_#3")), 'clientresult'] = 3
    labs.ix[((ctn) & (labs.clientresult == "tube_#4")), 'clientresult'] = 4

    rii = labs.description == 'rmsf,igg,ifa'
    labs.ix[((rii) & (labs.clientresult == "1:64")), 'clientresult'] = 1
    labs.ix[((rii) & (labs.clientresult == "1:128")), 'clientresult'] = 2
    labs.ix[((rii) & (labs.clientresult == "1:256")), 'clientresult'] = 3

    rr = labs.description == 'reference_ranges'
    labs.ix[((rr) & (labs.clientresult == "art_ref_range")), 'clientresult'] = 1
    labs.ix[((rr) & (labs.clientresult == "ven_ref_range")), 'clientresult'] = 2

    labs.ix[labs.clientresult == 'massive', 'clientresult'] = 5
    labs.ix[labs.clientresult == 'many', 'clientresult'] = 4

    sta = labs.description == 'fluid_appearance'
    labs.ix[((sta) & (labs.clientresult == "clear")), 'clientresult'] = 0
    labs.ix[((sta) & (labs.clientresult == "opaque")), 'clientresult'] = 0.5
    labs.ix[((sta) & (labs.clientresult == "milky")), 'clientresult'] = 1
    labs.ix[((sta) & (labs.clientresult == "hazy")), 'clientresult'] = 1.5
    labs.ix[((sta) & (labs.clientresult == "grossly_bloody")), 'clientresult'] = 2
    labs.ix[((sta) & (labs.clientresult == "cloudy")), 'clientresult'] = 2.5
    labs.ix[((sta) & (labs.clientresult == "bloody")), 'clientresult'] = 3
    labs.ix[((sta) & (labs.clientresult == "turbid")), 'clientresult'] = 3.5
    labs.ix[((sta) & (labs.clientresult == "slightly_cloudy")), 'clientresult'] = 4
    labs.ix[((sta) & (labs.clientresult == "blood_tinged")), 'clientresult'] = 4.5

    col = labs.description == 'fluid_color'
    labs.ix[((col) & (labs.clientresult == "colorless")), 'clientresult'] = 0
    labs.ix[((col) & (labs.clientresult == "red")), 'clientresult'] = 0.5
    labs.ix[((col) & (labs.clientresult == "xanthochromic")), 'clientresult'] = 1
    labs.ix[((col) & (labs.clientresult == "light_pink")), 'clientresult'] = 1.5
    labs.ix[((col) & (labs.clientresult == "orange")), 'clientresult'] = 2
    labs.ix[((col) & (labs.clientresult == "brown")), 'clientresult'] = 2.5
    labs.ix[((col) & (labs.clientresult == "amber")), 'clientresult'] = 3
    labs.ix[((col) & (labs.clientresult == "yellow")), 'clientresult'] = 3.5
    labs.ix[((col) & (labs.clientresult == "light_yellow")), 'clientresult'] = 4
    labs.ix[((col) & (labs.clientresult == "pale_red")), 'clientresult'] = 4.5
    labs.ix[((col) & (labs.clientresult == "white")), 'clientresult'] = 4.5

    schi = labs.description == 'schistocytes'
    labs.ix[((schi) & (labs.clientresult == "several")), 'clientresult'] = 4

    intu = labs.description == 'intubated?_y/n'
    labs.ix[((intu) & (labs.clientresult == "yes")), 'clientresult'] = 1
    labs.ix[((intu) & (labs.clientresult == "no")), 'clientresult'] = 0

    wbcc = labs.description == 'urine_wbc_clumps'
    labs.ix[((wbcc) & (labs.clientresult == "occ")), 'clientresult'] = 0
    labs.ix[((wbcc) & (labs.clientresult == "few")), 'clientresult'] = 1
    labs.ix[((wbcc) & (labs.clientresult == "occasional")), 'clientresult'] = 2

    hept = labs.description == 'hepatitis_b_surface_antibody'
    labs.ix[((hept) & (labs.clientresult == "non_reactive")), 'clientresult'] = 0
    labs.ix[((hept) & (labs.clientresult == "reactive")), 'clientresult'] = 1

    ty = labs.description == 'rheumatoid_factor'
    labs.ix[((ty) & (labs.clientresult == "negative")), 'clientresult'] = 50
    labs.ix[((ty) & (labs.clientresult == "positive")), 'clientresult'] = 70


    return labs


def pro_labs(dire=PROPATH, save=PROPATH):
    """
    Process labs_remove.csv
    Drop chstandard (62% null values)
    Drop clientresult canceled
    Correct gfr columns
    Correct pot columns
    Correct categorical columns
    Correct clientresults
    """
    labs = pd.read_csv(get_path(dire, 'labs_remove.csv'))

    del labs['chstandard']
    labs = labs[(labs.clientresult != 'canceled')]
    labs = labs[(labs.description.str[-1] != '$')]

    labs = _pro_gfr(labs)
    labs = _pro_cat(labs)
    labs = _pro_clean_clientresults(labs)

    if save:
        labs.to_csv(get_path(save, 'labs.csv'), index=False)

    return labs


def regen_labs_data(dire=PROPATH):
    pro_labs_basic(dire, dire)
    remove_desc(dire, dire)
    labs = pro_labs(dire, dire)

    return labs


def check_desc(fname, start=None, to=None, dire=PROPATH):
    """
    checks labs.csv for uncleaned descriptions
    """
    not_cleaned, checked = [], []
    labs = pd.read_csv(get_path(dire, 'labs.csv'))

    with open(fname) as f:
        for i, line in enumerate(f.readlines()):
            num = i + 1
            if to and num > to:
                break
            if start == None or num >= start:
                d = line.split()[0]
                checked.append(d)
                try:
                    labs[labs.description == d].clientresult.unique().astype(np.float64)
                except ValueError:
                    not_cleaned.append(d)

    return checked, not_cleaned


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
    regen_labs_data(save)


if __name__ == '__main__':
    process(*sys.argv[1:])
