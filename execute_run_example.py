import pickle
		
select_index = pickle.load(open("./tools/frontend/nasbench/model_index.pickle", "rb"))
print(len(select_index))

for i in range(len(select_index)):
	cmd = './maestro --HW_file="data/hw/accelerator_1.m" --Mapping_file="data/mapping/{}_out.m" --print_res=true --print_res_csv_file=true --print_log_file=false'.format(i)
	os.system(cmd)
	
	
import pandas as pd
import csv
import os
import pickle

latency = []
for i in range(len(select_index)):
	with open(str(i) + '_out.csv','r') as f:
		reader = csv.reader(f)
		rumtime_col = [row[3] for row in reader]	# column 3 is runtime
		tmp = [float(rumtime_col[j]) for j in range(1, len(rumtime_col))]
		#print(tmp)
		latency.append(sum(tmp))
		#print(latency)

pickle.dump(latency, open("acc1_latency.pickle", "wb"))
	
energy = []
for i in range(len(select_index)):
	with open(str(i) + '_out.csv','r') as f:
		reader = csv.reader(f)
		energy_col = [row[4] for row in reader]	# column 4 is energy
		tmp = [float(energy_col[j]) for j in range(1, len(energy_col))]
		#print(tmp)
		energy.append(sum(tmp))
		#print(energy)
		
pickle.dump(energy, open("acc1_energy.pickle", "wb"))