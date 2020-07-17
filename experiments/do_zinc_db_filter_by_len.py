import numpy as np
import time
import os
import time
import pickle
import sys

import argparse
parser = argparse.ArgumentParser(description='Run smiles extraction')

parser.add_argument('-i','--input_dir', type=str, help='Path to the input data', required=True)
parser.add_argument('-o','--output_dir', type=str, help='Path to the output data', required=True)

args = vars(parser.parse_args())
zinc_path = args['input_dir']+"/"
output_dir = args['output_dir']+"/"

zinc_dirs = [l for l in os.listdir(zinc_path) if ".wget" not in l]

for zd in zinc_dirs:
    st = time.time()
    zinc_smiles_set = []
    zinc_mol_set = []
    zinc_files = os.listdir(zinc_path+zd)
    for zf in zinc_files:
        f_object = open(zinc_path+zd+"/"+zf,"r")
        f_object.readline()
        for zinc_index,lines in enumerate(f_object):
            zinc_smiles = lines.split(" ")[0]
            #calculate the smiles length
            try:
                zinc_length = len(zinc_smiles)
            except:
                print("skipped smiles: ",zinc_smiles)
            #filter string length
            if (zinc_length < 110)*(zinc_length > 15):
                zinc_smiles_set.append(zinc_smiles)

        f_object.close()
    print(len(zinc_smiles_set))
    print("dir "+zd+" took ",time.time()-st,"seconds")
    print(output_dir+zd+"_processed.npy")
    output_file = open(output_dir+"/"+zd+"_processed.pkl","+wb")
    pickle.dump(zinc_smiles_set,output_file)
