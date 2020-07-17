import numpy as np
import pickle
import os

import argparse
parser = argparse.ArgumentParser(description='Combine distance rank data sets')

parser.add_argument('-i','--ranked_dir', type=str, help='Input directory', required=True)
parser.add_argument('-o','--output_file', type=str, help='output_file', required=True)
parser.add_argument('-all_smi','--all_smi', type=str, help='all_smi.txt file', required=True)
parser.add_argument('-n_all_smi','--n_all_smi', type=int, help='number of entries in all_smi.txt', required=True)
parser.add_argument('-small_smi','--small_smi', type=str, help='small_smi.txt file', required=True)
parser.add_argument('-n_small_smi','--n_small_smi', type=int, help='number of entries in small_smi.txt', required=True)

args = vars(parser.parse_args())

ranked_data_dir = args['ranked_dir']+"/"
databases = [pickle.load(open(ranked_data_dir+l,"rb")) for l in os.listdir(ranked_data_dir) if ".pkl" in l]

output_filename = args['output_file']
output_file = open(args['output_file'],"w+")
output_file.write("SMILES,Distance\n")

all_smi_filename = args['all_smi']
all_smile = open(all_smi_filename,"w+")

small_smi_filename = args['small_smi']
small = open(small_smi_filename,"w+")

n_all_smi = args['n_all_smi']
n_small_smi = args['n_small_smi']

combined_data = [[],[]]

for db in databases:
    combined_data = np.hstack([combined_data,db])

combined_data = np.asarray(combined_data)
sort_index = np.argsort(np.asarray(combined_data[1],dtype=float))[:n_all_smi]
combined_data = [combined_data[0][sort_index],combined_data[1][sort_index]]

for index in range(len(combined_data[0])):
    output_file.write(combined_data[0][index]+","+str(combined_data[1][index])+"\n")
    all_smile.write(combined_data[0][index]+"\n")
    if index < n_small_smi:
        small.write(combined_data[0][index]+"\n")

output_file.close()
all_smile.close()
small.close()
