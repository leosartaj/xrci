from StringIO import StringIO
import numpy as np
import pandas as pd

import pre, fea


def read_lists(li, header=''):
    for i, l in enumerate(li):
        li[i] = ','.join(l)

    s = StringIO('\n'.join([header, '\n'.join(li)]))
    return pd.read_csv(s)


def vitals_features(vital_list):
    vital_header = 'id,timestamp,measure,result,icu,seq'

    vitals = read_lists(vital_list, vital_header)

    vitals['measure'] = vitals['measure'].map(lambda x: x.lower().
                                             replace(' ', '_'))

    vp = vitals.pivot(columns='measure', values='result')
    vp['id'] = vitals.id
    #vp['icu'] = vitals.icu
    vp['timestamp'] = vitals.timestamp
    vp.columns = np.array(vp.columns)
    vpg = vp.groupby(['id', 'timestamp'], as_index=False).mean()

    return vpg


def static_featues(static_list):
    static_header = 'id,age,gender,maritalstatus,ethnicgroup,admitspecialty'

    s = StringIO('\n'.join([static_header, ','.join(static_list)]))
    static = pd.read_csv(s)
    static.columns = [col.lower() for col in static.columns]
    static = static.applymap(lambda x: x.lower().replace(' ', '_').
                             replace('-', '_') if isinstance(x, str) else x)
    static.ix[(static.gender == 'm'), 'gender'] = 0
    static.ix[(static.gender == 'f'), 'gender'] = 1

    return static


def clean_labs(labs_list):
    labs_header = 'id,timestamp,chstandard,clientresult,description,observationdescription,unitofmeasure,seq'
    labs = read_lists(labs_list, labs_header)

    del labs['chstandard']
    del labs['observationdescription']
    del labs['unitofmeasure']
    del labs['seq']

    labs['description'] = labs['description'].map(lambda x: x.lower().
                                                    replace(' ', '_'))

    labs['clientresult'] = (labs['clientresult'].
                                      map(lambda x: x.lower().replace(' ', '_')
                                          if not x is np.nan else x))
    labs = labs.drop_duplicates()

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

    labs = labs.reset_index()
    del labs['index']

    labs = labs[(labs.clientresult != 'canceled')]
    labs = labs[(labs.description.str[-1] != '$')]

    labs = pre._pro_gfr(labs)
    labs = pre._pro_fewdesccorrections(labs)
    labs = pre._pro_cat(labs)
    labs = pre._pro_clean_clientresults(labs)

    return labs


def labs_features(labs, vitals, static, dis):
    features, r_labs = fea.get_features_ranges(dis + '_feature.txt')

    query = labs.description == features[0]

    for f in features[1:]:
        query = query | (labs.description == f)

    labs = labs[query]

    labs.clientresult = labs.clientresult.astype(np.float64)

    lp = labs.pivot(columns='description', values='clientresult')
    lp['id'] = labs.id
    lp['timestamp'] = labs.timestamp
    lp.columns = np.array(lp.columns)
    lpg = lp.groupby(['id', 'timestamp'], as_index=False).mean()

    for f in features:
        if not f in lpg.columns:
            lpg[f] = np.nan

    features_v, r_vitals = fea.get_features_ranges('vitals.txt')
    lpg = fea.merge_df(vitals, r_vitals, lpg, r_labs)

    cols = list(features)
    cols.extend(features_v)
    lpg = fea.cummean(lpg, cols) # can be changed

    static = static[['id', 'age', 'gender']]
    lpg = pd.merge(lpg, static, on='id')

    return lpg.iloc[:, 2:]


def get_feature_set(vital_list, labs_list, static_list):
    vitals = vitals_features(vital_list)
    static = static_featues(static_list)
    labs = clean_labs(labs_list)
    dis = ['pne']
    features = []
    for dis in diseases:
        features.append(labs_features(labs.copy(), vitals.copy(), static.copy(), dis))

    return features
