import numpy as np
import time
import os
import time
import pickle
import sys
from rdkit.Chem import MolFromSmiles as MS
from rdkit.Chem import MolToSmiles as SM
from rdkit import Chem
from math import sqrt
from joblib import Parallel, delayed
import rdkit
from rdkit.Chem import RDConfig
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

sys.path.append('../src')
import fixed_parameters as fp
import helper as hp

d_vocab_smi = fp.TOKEN_INDICES_SMI

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

import argparse
parser = argparse.ArgumentParser(description='Run distance rank')

parser.add_argument('-i','--input', type=str, help='Input file', required=True)
parser.add_argument('-o','--output', type=str, help='Output file', required=True)
parser.add_argument('-n','--top_n', type=int, help='Top n most similar', required=True)
parser.add_argument('-t','--template_file', type=str, help='Path to the template file', required=True)

args = vars(parser.parse_args())
input_filename = args['input']
output_filename = args['output']
top = args['top_n']

def tanimoto_dist(A,B):
    return 1-rdkit.DataStructs.FingerprintSimilarity(A,B,metric=rdkit.DataStructs.TanimotoSimilarity)

def convert_to_morgan(mol):
    morgan = GetMorganFingerprintAsBitVect(mol,2,nBits=1024)
    return morgan

def ave_morgan_dist(mol,template_desc):
    template1 = template_desc[0]
    template2 = template_desc[1]

    dist_output = []
    mol_desc = convert_to_morgan(mol)
    dist_val1 = tanimoto_dist(mol_desc,template1)
    dist_val2 = tanimoto_dist(mol_desc,template2)
    ave_dist = np.average([dist_val1,dist_val2])
    dist_output.append(ave_dist)
    return dist_output

template_file = open(args['template_file'],"r")
template_desc_list = [convert_to_morgan(MS(template.strip("\n"))) for template in template_file]
template_file.close()

f_object = open(input_filename,"r")
mol_set = []
dist_set = []

combined_data = [[],[]]
for index,line in enumerate(f_object):
    if index % 10**3 == 0:
        print(index)
    try:
        smiles = line.strip("\n")
        #filter by uncanonicalized length
        if (len(smiles) < 110)*(len(smiles) > 15):
            mol = MS(smiles)
            #filter by SAScore
            if sascorer.calculateScore(mol) < 4.2:
                canon_smiles = SM(mol)
                #filter by canonicalized length
                if len(canon_smiles) < 100 and len(canon_smiles) > 25:
                    #check char within scope of the model
                    num_non_overlap = len(set(hp.smi_tokenizer(canon_smiles)).difference(d_vocab_smi.keys()))
                    if num_non_overlap == 0:
                        dist_val = ave_morgan_dist(mol,template_desc_list)[0]
                        mol_set.append(canon_smiles)
                        dist_set.append(dist_val)
    except:
        print(line)
    if index % 10**5 == 0:
        combined_data = np.hstack([combined_data,[mol_set,dist_set]])
        sorted_index = np.argsort(np.asarray(dist_set,dtype=float))[:10**5]
        combined_data = [combined_data[0][sorted_index],combined_data[1][sorted_index]]
        mol_set = []
        dist_set = []

combined_data = np.hstack([combined_data,[mol_set,dist_set]])
sorted_index = np.argsort(np.asarray(dist_set,dtype=float))[:top]
combined_data = [combined_data[0][sorted_index],combined_data[1][sorted_index]]

#remove duplicates
unique_smiles, unique_index = np.unique(combined_data[0],return_index=True)
combined_data = [combined_data[0][unique_index],combined_data[1][unique_index]]

output_file = open(output_filename,"wb+")
pickle.dump(combined_data,output_file)
