import pickle
import os
import numpy as np
import sys
sys.path.append('../src')
import fixed_parameters as fp
import helper as hp

d_vocab_smi = fp.TOKEN_INDICES_SMI

import argparse
parser = argparse.ArgumentParser(description='Compile ZINC distance results')

parser.add_argument('-i','--dist_dir', type=str, help='Input directory', required=True)
parser.add_argument('-o','--output', type=str, help='Output file', required=True)
parser.add_argument('-smi','--smi_dir_path', type=str, help='Output file', required=True)
parser.add_argument('-n','--num_top_entries', type=int, help='number of top entries in the output', required=True)

args = vars(parser.parse_args())
dist_dir_path = args['dist_dir']+"/"
smi_dir = args['smi_dir_path']+"/"
output_filename = args['output']
top = args['num_top_entries']

dist_files = [l for l in os.listdir(dist_dir_path)]

dist_data = [[],[],[]]
for df in dist_files:
    dist_data = np.hstack([dist_data,pickle.load(open(dist_dir_path+df,"rb"))])

sort_index = np.argsort(np.asarray(dist_data[-1],dtype=float))[:top]
sorted_dist_data = [dist_data[0][sort_index][:top],dist_data[1][sort_index][:top],dist_data[2][sort_index][:top]]

unique_filename = list(set(sorted_dist_data[0]))

mol_dir = smi_dir
compiled_dist_data = [[],[]]
for uf_index, uf in enumerate(unique_filename):
    if uf_index % 1000 == 0:
        print((uf_index+1)/len(unique_filename))
    file_index = np.where(sorted_dist_data[0] == uf)[0]
    mol_index = np.asarray(sorted_dist_data[1][file_index],dtype=float)
    M = np.asarray(pickle.load(open(mol_dir+uf.replace("_mol_","_smiles_"),"rb")))
    mol_data = M[np.asarray(mol_index,dtype=int)]
    mol_dist = sorted_dist_data[-1][file_index]
    compiled_dist_data = np.hstack([compiled_dist_data,[mol_data,mol_dist]])

#check char tokens
stored_index = []
for compile_index,canon_smiles in enumerate(compiled_dist_data[0]):
    num_non_overlap = len(set(hp.smi_tokenizer(canon_smiles)).difference(d_vocab_smi.keys()))
    if num_non_overlap == 0:
        stored_index.append(compile_index)

compiled_dist_data = [compiled_dist_data[0][stored_index],compiled_dist_data[1][stored_index]]

#remove duplicates
unique_smiles, unique_index = np.unique(compiled_dist_data[0],return_index=True)
compiled_dist_data = [compiled_dist_data[0][unique_index],compiled_dist_data[1][unique_index]]

output_dir = open(output_filename,"wb+")
pickle.dump(compiled_dist_data,output_dir)
output_dir.close()
