from util import get_test_data,get_val,get_time_list,join_pred

from util_my import get_path
import train, test

import pandas as pd
import numpy as np

def train_model(train_dir_path):
        ret_parameters = {}

        # pne model
        pne_model, nor = train.get_model('pne_model', 'pne_nor')
        ret_parameters['pne'] = (pne_model, nor)

        #ret_parameters['pne'] = None
        ret_parameters['cao'] = None
        ret_parameters['seps'] = None
        ret_parameters['chf'] = None
        ret_parameters['hfu'] = None
        ret_parameters['ami'] = None
        ret_parameters['pulm_ei'] = None
        ret_parameters['sseps'] = None
        ret_parameters['sepshock'] = None
        ret_parameters['intes_infec'] = None
        ret_parameters['pneitus'] = None
        ret_parameters['dhf'] = None
        ret_parameters['shf'] = None
        ret_parameters['sub_infrac'] = None
        ret_parameters['bas'] = None
        ret_parameters['ischemic_cd'] = None

	return ret_parameters

def predict(trained_model_parameters,id_,time_,vit_data_list,lab_data_list,rx_data_list,stat):
	#print "------------------------Uncomment below to see what type of data you are getting-----------------------"
	#print id_
	#raw_input()
	#print time_
        #raw_input()
        #print stat
        #raw_input()
        #print vit_data_list
        #raw_input()
        #print lab_data_list
        #raw_input()
        #print rx_data_list
        #raw_input()
        #print "-----------------------------------------------"

        #diseases=['486','434.91','995.91','428.0','428.9','410.9','415.19','995.92','785.52','008.45','507.0','428.30','428.20','410.7','435','437.']
        diseases=['pne','cao','seps','chf','hfu','ami','pulm_ei','sseps','sepshock','intes_infec','pneitus','dhf','shf','sub_infrac','bas','ischemic_cd']

        #example 1: if you predict negative/no complications for all diseases then your return list will look like
        #ret=[id_]+['NA']*16

        #example 2: if you predict positive for  2nd(434.91), 4th(428.0) and 12th(428.30) complications and negative for others then your return list will be
        #ret=[id_,'NA',time_,'NA',time_,'NA','NA','NA','NA','NA','NA','NA',time_,'NA','NA','NA','NA']

        ret=[id_]

        features = test.get_feature_set(vit_data_list, lab_data_list, stat)

        for dis in diseases:
            model = trained_model_parameters[dis]
            if model:
                df = features.get(dis, None)
                if df is not None:
                    n = model[1]
                    for c in df.columns:
                        df.ix[:, c] = df.ix[:, c].apply(lambda x: (x - n[c][0]) / n[c][1])
                    x = np.array(df)
                    pred = model[0].predict(x).sum()
                    if pred > 0:
                        ret.append(time_)
                    else:
                        ret.append('NA')
                else:
                    ret.append('NA')
            else:
                ret.append('NA')

	return ret

def check_pred(pred,id_,time_):
	new_pred=[]
	for i in pred:
		try:
			c=int(float(i))
			if c not in [id_,time_]:

				print "Error: your returned pred has elements other than id_=",id_,"time_=",time_, " or 'NA'"
				print "your returned pred is ",pred
				return False,[]
			new_pred.append(c)
		except ValueError:
			if i!='NA':
				print "Error: your returned pred has elements other than id_=",id_,"time_=",time_, " or 'NA'"
				print "your pred is ",pred
				return False,[]
			new_pred.append(i)
	assert(len(pred)==17)
	return pred,new_pred


def predict_online(trained_model_parameters,test_data):
	#unfold test_data
	dict_vital_lol,dict_vital_time,id_list,dict_lab_lol,dict_lab_time,dict_static_data,dict_Rx_lol,dict_Rx_time=test_data
	header=['id','486','434.91','995.91','428.0','428.9','410.9','415.19','995.92','785.52','008.45','507.0','428.30','428.20','410.7','435','437.']
	assert(len(header)==17)
	all_output=[header]
	print "online prediction started"
	for id_ in id_list:
		output=[]
		vit_data_list=get_val(id_,dict_vital_lol)
		lab_data_list=get_val(id_,dict_lab_lol)
		rx_data_list=get_val(id_,dict_Rx_lol)
		time_list,vit_time_list,lab_time_list,rx_time_list=get_time_list(id_,dict_vital_time,dict_lab_time,dict_Rx_time)
		assert(len(vit_data_list)==len(vit_time_list) and len(lab_data_list)==len(lab_time_list) and len(rx_data_list)==len(rx_time_list))
		vit,lab,rx=[],[],[]
		stat=get_val(id_,dict_static_data)
		stat=stat if len(stat)!=0 else ['NA']*6
		for time_ in time_list:
			vit_ind=[i for i in range(len(vit_time_list)) if vit_time_list[i] > time_]
			vit_ind=vit_ind[0] if len(vit_ind)!=0 else len(vit_time_list)
			lab_ind=[ i for i in range(len(lab_time_list)) if lab_time_list[i]> time_]
			lab_ind=lab_ind[0] if len(lab_ind)!=0 else len(lab_time_list)
			rx_ind=[i for i in range(len(rx_time_list)) if rx_time_list[i]> time_]
			rx_ind=rx_ind[0] if len(rx_ind)!=0 else len(rx_time_list)
			#Participants have to populate predict function called
			#input: id_, time_, vit_data_list= list of list of vital data till time_, similarly others
			#output: single list having 17 elements.
			#first element is id_ and remaining data is prediction for 16 complications '486','434.91','995.91','428.0','428.9','410.9','415.19','995.92','785.52','008.45','507.0','428.30','428.20','410.7','435','437.']
			#note: if you predict positive for any complication then instead of 1 at the returning list place the time_ value( this was passed to you during call of predict function). if you predict 0 then put 'NA' in the list
			pred=predict(trained_model_parameters,id_,time_,vit_data_list[:vit_ind],lab_data_list[:lab_ind],rx_data_list[:rx_ind],stat)

			check,new_pred=check_pred(pred,id_,time_)
			if check==False:
				print("note: predicted list returned from predict function should not have elements other than with id_,time_ or 'NA'")
				import sys
				sys.exit(0)
			assert(len(pred)==17)
			output.append(new_pred)
		all_output.append(join_pred(id_,output))
	return all_output


import sys
if len(sys.argv)!=4:
	print "Usage Error: python generate_output_csv.py train_dir_path test_dir_path outputfilename"
	print "note: test and train dir paths should not end with slash"
	sys.exit(0)

train_dir_path=sys.argv[1];test_dir_path=sys.argv[2];outputfilename=sys.argv[3]
print "train_dir_path=",train_dir_path,"  test_dir_path=",test_dir_path

test_fnames=["test_RawVitalData.csv","test_RawLabData.csv","test_Rxdata.csv","test_Static_data.csv"]
print "Note: Test filenames in test_folder_path should be ",test_fnames
sep="/"

#participants have to call their function inside below function to train model and return parameters(such as model parameters, mean etc) needed for testing
trained_model_parameters=train_model(train_dir_path)

#Participants can treat this function as black box. this function gets test data dictionaries-
test_data= get_test_data(test_dir_path,sep,test_fnames)

#Inside this function participants have to populate predict function treating all the other content as block box. This function does online prediction
output=predict_online(trained_model_parameters,test_data)

#write output
import csv
with open(outputfilename,"wb") as f:
	writer=csv.writer(f)
	writer.writerows(output)
