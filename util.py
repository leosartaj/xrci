
def join_pred(id_,output):
	#print "inside join pred",output
	#raw_input()
	temp=[id_]+['NA']*16
	for row in output:
		na_indices=[i_ for i_ in range(len(temp)) if temp[i_]=='NA']
		if len(na_indices)==0:
			return temp
		for i_ in na_indices:
			if row[i_] !='NA':
				temp[i_]=row[i_]
	
	#print "inside join pred",temp
	#raw_input()
	return temp


def get_time_list(id_,dict_vital_time,dict_lab_time,dict_Rx_time):
	vit_time_list=get_val(id_,dict_vital_time)
	if len(vit_time_list)==0:
		print "error: vitals data missing for patient id",id_
		print "program stopped at get_time_list function of generate_output.py"
		import sys
		sys.exit(0)
	lab_time_list=get_val(id_,dict_lab_time)
	rx_time_list=get_val(id_,dict_Rx_time)
	new_time_list=sorted(list(set(vit_time_list+[val for val in lab_time_list if val>vit_time_list[-1]]+[val for val in rx_time_list if val>vit_time_list[-1]])))
	return new_time_list,vit_time_list,lab_time_list,rx_time_list


def get_val(id_,dict_):
	try:
		list_=dict_[id_]
	except KeyError:
		list_=[]
	return list_

def generate_vital_data(vital_fname):
	f=open(vital_fname)
	import csv
	reader=csv.reader(f)
	data=[]
	header=True
	for row in reader:
		if header==True:
			header=False
			continue
		data.append(row)
	f.close()
	dict_vital_lol={}
	dict_vital_time={}
	id_list=[]
	for m in data:
		try:
			dict_vital_lol[int(float(m[0]))].append(m)
			dict_vital_time[int(float(m[0]))].append(int(float(m[1])))
		except KeyError:
			dict_vital_lol[int(float(m[0]))]=[m]
			id_list.append(int(float(m[0])))
			dict_vital_time[int(float(m[0]))]=[int(float(m[1]))]
	#print "check: id_list before sending",len(id_list),len(set(id_list))
	return dict_vital_lol,dict_vital_time,list(set(id_list))

def generate_lab_data(lab_fname):
	f=open(lab_fname)
	import csv
	reader=csv.reader(f)
	data=[]
	header=True
	for row in reader:
		if header==True:
			header=False
			continue
		data.append(row)
	f.close()
	dict_lab_lol={}
	dict_lab_time={}
	for m in data:
		try:
			dict_lab_lol[int(float(m[0]))].append(m)
			dict_lab_time[int(float(m[0]))].append(int(float(m[1])))
		except KeyError:
			dict_lab_lol[int(float(m[0]))]=[m]
			dict_lab_time[int(float(m[0]))]=[int(float(m[1]))]
	return dict_lab_lol,dict_lab_time
			
def generate_static_data(test_static_fname):
	f=open(test_static_fname)
	import csv
	reader=csv.reader(f)
	data=[]
	header=True
	for row in reader:
		if header==True:
			header=False
			continue
		data.append(row)
	f.close()
	dict_static_data={}
	for m in data:
		dict_static_data[int(float(m[0]))]=m
	return dict_static_data

def generate_Rx_data(Rx_fname):
	f=open(Rx_fname)
	import csv
	reader=csv.reader(f)
	data=[]
	
	header=True
	for row in reader:
		if header==True:
			header=False
			continue
		data.append(row)
	f.close()
	dict_Rx={}
	dict_Rx_time={}
	for m in data:
		try:
			dict_Rx[int(float(m[0]))].append(m)
			dict_Rx_time[int(float(m[0]))].append(int(float(m[1])))
		except KeyError:
			dict_Rx[int(float(m[0]))]=[m]
			dict_Rx_time[int(float(m[0]))]=[int(float(m[1]))]
	return dict_Rx,dict_Rx_time

def test_dict(dict_,str_):
	print "-----------------------"
	print str_
	print dict_.keys()[0]
	print dict_[dict_.keys()[0]]
	print "-----------------------"
	
def get_test_data(test_dir_path,sep,test_fnames):

	vit_fname=test_dir_path+sep+test_fnames[0]
	dict_vital_lol,dict_vital_time,id_list=generate_vital_data(vit_fname)
	#print "completed vital data generation len of id_list is",len(id_list), len(dict_vital_lol.keys()),len(dict_vital_time.keys())
	#test_dict(dict_vital_lol,"dict_vital_lol")
	#test_dict(dict_vital_time,"dict_vital_time")
	#raw_input()
	
	#get lab data
	lab_fname=test_dir_path+sep+test_fnames[1]
	dict_lab_lol,dict_lab_time=generate_lab_data(lab_fname)
	#print "completed lab data generation ",len(dict_lab_lol.keys()),len(dict_lab_time.keys())
	#test_dict(dict_lab_lol,"dict_lab_lol")
	#test_dict(dict_lab_time,"dict_lab_time")
	#raw_input()

	#get static data
	static_data_fname=test_dir_path+sep+test_fnames[3]
	dict_static_data=generate_static_data(static_data_fname)
	#print "completed static data generation",len(dict_static_data.keys())
	#test_dict(dict_static_data,"dict_static_data")
	#raw_input()
	#get medication data
	Rx_fname=test_dir_path+sep+test_fnames[2]
	dict_Rx_lol,dict_Rx_time=generate_Rx_data(Rx_fname)
	#print "completed medication data generation",len(dict_Rx_time.keys()),len(dict_Rx_lol.keys())
	#test_dict(dict_Rx_lol,"dict_rx_lol")
	#test_dict(dict_Rx_time,"dict_rx_time")
	#raw_input()

	return [dict_vital_lol,dict_vital_time,id_list,dict_lab_lol,dict_lab_time,dict_static_data,dict_Rx_lol,dict_Rx_time]
