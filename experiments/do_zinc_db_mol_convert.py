import numpy as np
import time
import os
import time
import pickle
import sys
from rdkit.Chem import MolFromSmiles as MS
from rdkit import Chem
from math import sqrt

sys.path.append('../')
from src import fixed_parameters as fp
from src import helper as hp

d_vocab_smi = fp.TOKEN_INDICES_SMI

import argparse
parser = argparse.ArgumentParser(description='Run mol conversion')

parser.add_argument('-i','--input_dir', type=str, help='Path to the input data', required=True)
parser.add_argument('-mol','--output_mol_dir', type=str, help='Path to the output mol data', required=True)
parser.add_argument('-smi','--output_smi_dir', type=str, help='Path to the output smiles data', required=True)
parser.add_argument('-s','--split_index', type=int, help='Split index', required=True)
parser.add_argument('-n','--split_num', type=int, help='Split number', required=True)

args = vars(parser.parse_args())
zinc_path = args['input_dir']+"/"
zinc_files = np.sort([l for l in os.listdir(zinc_path) if ".wget" not in l])

output_dir_mol = args['output_smi_dir']+"/"
output_dir_smiles = args['output_smi_dir']+"/"

zinc_split_index = args['split_index']
split_num = args['split_num']

def convert_mol_filter_smiles(zinc_smiles):
    z_mol_list = []
    z_smiles_list = []
    for zinc_index,zs in enumerate(zinc_smiles):
        try:
            zinc_mol = Chem.MolFromSmiles(zs)
            canonic_zinc_smiles = Chem.MolToSmiles(zinc_mol)
            #check if the string contains characters outside of our dictionary
            num_non_overlap = len(set(hp.smi_tokenizer(canonic_zinc_smiles)).difference(d_vocab_smi.keys()))
            if num_non_overlap == 0:
                len_canonic = len(canonic_zinc_smiles)
                #filter canonical SMILES string length
                if len_canonic < 100 and len_canonic > 25:
                    z_mol_list.append(zinc_mol)
                    z_smiles_list.append(canonic_zinc_smiles)
        except:
            print("skipped smiles: ",zs)
    return z_mol_list, z_smiles_list

os.system("echo calc_time > zinc_"+str(zinc_split_index)+".out")

for zf_index,zf in enumerate(np.array_split(zinc_files,split_num)[zinc_split_index]):
    zinc_file = open(zinc_path+zf,"rb")
    zinc_smiles_pickle = pickle.load(zinc_file)
    zinc_file.close()

    logging_partition=10**5
    if len(zinc_smiles_pickle) != 0:
        if len(zinc_smiles_pickle) > logging_partition:
            split_num = len(zinc_smiles_pickle)/logging_partition
            zf_pickle_split = np.array_split(range(len(zinc_smiles_pickle)),split_num)
        else:
            zf_pickle_split = [range(len(zinc_smiles_pickle))]
        for zf_ps_index,zf_ps in enumerate(zf_pickle_split):
            st = time.time()
            zinc_canon_mol,zinc_canon_smiles = convert_mol_filter_smiles(zinc_smiles_pickle[zf_ps[0]:zf_ps[-1]])
            output_line = "file "+zf+" "+str(zf_index)+" took "+str(time.time()-st)+" seconds for "+str(len(zinc_canon_mol))+" mols"
            os.system("echo "+output_line+" >> zinc_"+str(zinc_split_index)+".out")
            zinc_canon_mol = list(filter(None.__ne__, zinc_canon_mol)) #remove None entry
            zinc_canon_smiles = list(filter(None.__ne__, zinc_canon_smiles)) #remove None entry
            #saves the mol outputs
            output_file = open(output_dir_mol+zf.replace(".pkl","_"+str(zf_index)+"_mol_canon.pkl"),"+wb")
            pickle.dump(zinc_canon_mol,output_file)
            output_file.close()
            #saves the smiles outputs
            output_file = open(output_dir_smiles+zf.replace(".pkl","_"+str(zf_index)+"_smiles_canon.pkl"),"+wb")
            pickle.dump(zinc_canon_smiles,output_file)
            output_file.close()
os.system("echo calc finished >> zinc_"+str(zinc_split_index)+".out")
print("calculation finished")
