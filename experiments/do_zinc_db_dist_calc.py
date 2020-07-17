import numpy as np
import time
import os
import time
import pickle
import sys
from rdkit.Chem import MolFromSmiles as MS
from rdkit import Chem
from math import sqrt
from joblib import Parallel, delayed
import rdkit
from rdkit.Chem import RDConfig
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

import argparse
parser = argparse.ArgumentParser(description='Run smiles extraction')

parser.add_argument('-i','--input_dir', type=str, help='Path to the input mol data', required=True)
parser.add_argument('-o','--output_dir', type=str, help='Path to the output directory', required=True)
parser.add_argument('-t','--template_file', type=str, help='Path to the template file', required=True)
parser.add_argument('-s','--split_index', type=int, help='Split index', required=True)
parser.add_argument('-n','--split_num', type=int, help='Split number', required=True)
parser.add_argument('-l','--log_num', type=int, help='Top l most similar compounds', required=True)

args = vars(parser.parse_args())
zinc_path = args['input_dir']+"/"
zinc_files = np.sort([l for l in os.listdir(zinc_path) if ".wget" not in l])

output_dir_dist = args['output_dir']+"/"

zinc_split_index = args['split_index']
split_num = args['split_num']
top_ranking = args['log_num']

def tanimoto_dist(A,B):
    return 1-rdkit.DataStructs.FingerprintSimilarity(A,B,metric=rdkit.DataStructs.TanimotoSimilarity)

def convert_to_morgan(mol):
    morgan = GetMorganFingerprintAsBitVect(mol,2,nBits=1024)
    return morgan

def morgan_dist(mol_list,template_desc):
    template1 = template_desc[0]
    template2 = template_desc[1]

    zinc_dist_list = []
    for mol in mol_list:
        try:
            mol_desc = convert_to_morgan(mol)
            dist_val1 = tanimoto_dist(mol_desc,template1)
            dist_val2 = tanimoto_dist(mol_desc,template2)
            ave_dist = np.average([dist_val1,dist_val2])
            zinc_dist_list.append(ave_dist)
        except:
            #desc_func(mol)
            print("skipped smiles: ")
            pass
    return zinc_dist_list

template_file = open(args['template_file'],"r")
template_desc_list = [convert_to_morgan(MS(template.strip("\n"))) for template in template_file]
template_file.close()

print("inputs",zinc_split_index, split_num)
os.system("echo calc_time > zinc_"+str(zinc_split_index)+".out")

#first calculation
sorted_dist = []
sorted_index = []
sorted_file_label = []

for zf_index,zf in enumerate(np.array_split(zinc_files,split_num)[zinc_split_index]):
    st = time.time()

    zinc_file = open(zinc_path+zf,"rb")
    zinc_mol_pickle = pickle.load(zinc_file)
    zinc_file.close()

    zinc_dist_list = morgan_dist(zinc_mol_pickle,template_desc_list)
    index_list = list(range(len(zinc_dist_list)))
    file_label_list = [zf for i in range(len(zinc_dist_list))]

    #calculate the sort index
    top_indices = np.argsort(np.concatenate([sorted_dist,zinc_dist_list]))[:top_ranking]

    #sorted dist output data
    sorted_dist = np.concatenate([sorted_dist,zinc_dist_list])[top_indices]
    sorted_index = np.concatenate([sorted_index,index_list])[top_indices]
    sorted_file_label = np.concatenate([sorted_file_label,file_label_list])[top_indices]

    output_line = "file "+zf+" took "+str(time.time()-st)+" seconds for "+str(len(zinc_dist_list))+" mols"
    os.system("echo "+output_line+" >> zinc_"+str(zinc_split_index)+".out")
    zinc_dist_output = list(filter(None.__ne__, zinc_dist_list)) #remove None entry
    #save the zinc mol objects
output_file = open(output_dir_dist+"Zinc_ave_morgan_dist_top_rank_SPLIT-"+str(zinc_split_index)+".pkl","+wb")
pickle.dump([sorted_file_label,sorted_index,sorted_dist],output_file)
output_file.close()

os.system("echo calc finished >> zinc_"+str(zinc_split_index)+".out")
print("calculation finished")
