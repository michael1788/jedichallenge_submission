import time
import os
import rdkit
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import MolFromSmiles as MS
from rdkit.Chem import MolToSmiles as SM
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem as Chem
import time
from functools import partial
import pickle
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint
from rdkit.Chem.rdMolDescriptors import GetHashedAtomPairFingerprintAsBitVect as AtomPair
from rdkit.Chem.rdMolDescriptors import GetHashedTopologicalTorsionFingerprintAsBitVect as Tor
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem.rdmolops import LayeredFingerprint
from rdkit.Chem.rdmolops import RDKFingerprint
from rdkit.Chem.Crippen import MolLogP
import itertools
from scipy.stats import spearmanr

from scipy.spatial.distance import euclidean
from scipy.spatial.distance import rogerstanimoto

cats_path="../src/CATS2D"
import sys
sys.path.append(cats_path)
from cats2d.rd_cats2d import CATS2D
cats = CATS2D()

import argparse
parser = argparse.ArgumentParser(description='Compare descriptor ranking')

parser.add_argument('-i','--input_file', type=str, help='Input file', required=True)
parser.add_argument('-o','--output_file', type=str, help='Output file', required=True)
parser.add_argument('-t','--template_file', type=str, help='Path to the template file', required=True)
parser.add_argument('-m','--measurement', type=str, help='Measurement field to use', required=True)

#Distance metrics
#######################
def tanimoto_dist(A,B):
    return rogerstanimoto(A,B)

def compare_diff(A,B):
    return abs(A-B)
#######################

#Descriptors
#######################
def convert_to_maccs(SMILES):
    mol = MS(SMILES)
    maccs = Chem.GetMACCSKeysFingerprint(mol)
    maccs_float_list = list(np.asarray(list(maccs.ToBitString()),dtype=float))
    return maccs_float_list

def convert_to_MW(SMILES):
    mol = MS(SMILES)
    MW = Chem.Descriptors.ExactMolWt(mol)
    return MW

def convert_to_clogp(SMILES):
    mol = MS(SMILES)
    logp = MolLogP(mol)
    return logp

def convert_to_atompair(SMILES):
    mol = MS(SMILES)
    atom_pair = AtomPair(mol)
    atom_pair_float_list = list(np.asarray(list(atom_pair.ToBitString()),dtype=float))
    return atom_pair_float_list

def convert_to_tor(SMILES):
    mol = MS(SMILES)
    desc_val = Tor(mol)
    desc_val_float_list = list(np.asarray(list(desc_val.ToBitString()),dtype=float))
    return desc_val_float_list

def convert_to_avalon(SMILES):
    mol = MS(SMILES)
    desc_val = GetAvalonFP(mol)
    desc_val_float_list = list(np.asarray(list(desc_val.ToBitString()),dtype=float))
    return desc_val_float_list

def convert_to_morgan(SMILES):
    mol = MS(SMILES)
    morgan = Chem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024)
    morgan_float_list = list(np.asarray(list(morgan.ToBitString()),dtype=float))
    return morgan_float_list

def convert_to_featmorgan(SMILES):
    mol = MS(SMILES)
    morgan = Chem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024,useFeatures=True)
    morgan_float_list = list(np.asarray(list(morgan.ToBitString()),dtype=float))
    return morgan_float_list

def convert_to_layered(SMILES):
    mol = MS(SMILES)
    desc_val = LayeredFingerprint(mol)
    desc_val_float_list = list(np.asarray(list(desc_val.ToBitString()),dtype=float))
    return desc_val_float_list

def convert_to_rdkit(SMILES):
    mol = MS(SMILES)
    desc_val = RDKFingerprint(mol)
    desc_val_float_list = list(np.asarray(list(desc_val.ToBitString()),dtype=float))
    return desc_val_float_list

def convert_to_cats(SMILES):
    mol = MS(SMILES)
    cats_desc = cats.getCATs2D(mol)
    return cats_desc

def convert_to_HBA(SMILES):
    mol = MS(SMILES)
    val = rdkit.Chem.rdMolDescriptors.CalcNumHBA(mol)
    return val

def convert_to_HBD(SMILES):
    mol = MS(SMILES)
    val = rdkit.Chem.rdMolDescriptors.CalcNumHBD(mol)
    return val

def convert_to_num_aliphatic_hetero(SMILES):
    mol = MS(SMILES)
    val = rdkit.Chem.rdMolDescriptors.CalcNumAliphaticHeterocycles(mol)
    return val

def convert_to_num_aromatic_hetero(SMILES):
    mol = MS(SMILES)
    val = rdkit.Chem.rdMolDescriptors.CalcNumAromaticHeterocycles(mol)
    return val

def convert_to_num_rings(SMILES):
    mol = MS(SMILES)
    val = rdkit.Chem.rdMolDescriptors.CalcNumRings(mol)
    return val
#######################

#Descriptor dictionaries
#######################
MACCS_dict = {}
MACCS_dict['desc_func'] = convert_to_maccs
MACCS_dict['dist_func'] = tanimoto_dist
MACCS_dict['desc_name'] = "MACCS"

Morgan_dict = {}
Morgan_dict['desc_func'] = convert_to_morgan
Morgan_dict['dist_func'] = tanimoto_dist
Morgan_dict['desc_name'] = "Morgan"

CATS_dict = {}
CATS_dict['desc_func'] = convert_to_cats
CATS_dict['dist_func'] = euclidean
CATS_dict['desc_name'] = "CATS"

MW_dict = {}
MW_dict['desc_func'] = convert_to_MW
MW_dict['dist_func'] = compare_diff
MW_dict['desc_name'] = "MW"

clogp_dict = {}
clogp_dict['desc_func'] = convert_to_clogp
clogp_dict['dist_func'] = compare_diff
clogp_dict['desc_name'] = "ClogP"

featMorgan_dict = {}
featMorgan_dict['desc_func'] = convert_to_featmorgan
featMorgan_dict['dist_func'] = tanimoto_dist
featMorgan_dict['desc_name'] = "featMorgan"

Layered_dict = {}
Layered_dict['desc_func'] = convert_to_layered
Layered_dict['dist_func'] = tanimoto_dist
Layered_dict['desc_name'] = "Layered"

Atompair_dict = {}
Atompair_dict['desc_func'] = convert_to_atompair
Atompair_dict['dist_func'] = tanimoto_dist
Atompair_dict['desc_name'] = "AtomPair"

Avalon_dict = {}
Avalon_dict['desc_func'] = convert_to_avalon
Avalon_dict['dist_func'] = tanimoto_dist
Avalon_dict['desc_name'] = "Avalon"

Tor_dict = {}
Tor_dict['desc_func'] = convert_to_tor
Tor_dict['dist_func'] = tanimoto_dist
Tor_dict['desc_name'] = "Tor"

HBA_dict = {}
HBA_dict['desc_func'] = convert_to_HBA
HBA_dict['dist_func'] = compare_diff
HBA_dict['desc_name'] = "HBA"

HBD_dict = {}
HBD_dict['desc_func'] = convert_to_HBD
HBD_dict['dist_func'] = compare_diff
HBD_dict['desc_name'] = "HBD"

num_ali_hetero_dict = {}
num_ali_hetero_dict['desc_func'] = convert_to_num_aliphatic_hetero
num_ali_hetero_dict['dist_func'] = compare_diff
num_ali_hetero_dict['desc_name'] = "hetero_ali"

num_aro_hetero_dict = {}
num_aro_hetero_dict['desc_func'] = convert_to_num_aromatic_hetero
num_aro_hetero_dict['dist_func'] = compare_diff
num_aro_hetero_dict['desc_name'] = "hetero_arom"
#######################

if __name__ == '__main__':
    args = vars(parser.parse_args())
    input_filename = args['input_file']
    output_filename = args['output_file']

    #activity measurements
    measurement = args['measurement']
    #template smiles
    template_file = open(args['template_file'],"r")
    template_list = [template.strip("\n") for template in template_file]
    template_file.close()

    #Descriptors to measure
    rank_list = [MACCS_dict,
                 Morgan_dict,
                 CATS_dict,
                 MW_dict,
                 clogp_dict,
                 featMorgan_dict,
                 Atompair_dict,
                 Avalon_dict,
                 Layered_dict,
                 HBA_dict,
                 HBD_dict,
                 Tor_dict,
                 num_ali_hetero_dict,
                 num_aro_hetero_dict]


    #extract data
    f_object = open(input_filename,"r")
    activity_data = []
    activity_data_header = f_object.readline().strip("\n").split(",")

    measurement_index = [i for i in range(len(activity_data_header)) if activity_data_header[i] == measurement][0]
    for line in f_object:
        try:
            activity_entry = line.strip("\n").split(",")
            activity_data.append([activity_entry[0],float(activity_entry[measurement_index])])
        except:
            continue
    f_object.close()
    activity_data = np.asarray(activity_data)

    #perform ranking
    rank_output = []
    for rank_dict in rank_list:
        desc_func = rank_dict['desc_func']
        dist_func = rank_dict['dist_func']

        template_desc_list = [desc_func(template) for template in template_list]

        output = []
        desc_output = []
        for index,s in enumerate(activity_data[:,0]):
            if float(activity_data[index][1]) > 99.9:
                continue
            try:
                mol_desc = desc_func(s)
                template_dist_list = [dist_func(template_desc,mol_desc) for template_desc in template_desc_list]
                ave_dist = np.average(template_dist_list)
                output.append([s,activity_data[index][1],ave_dist])
                desc_output.append([mol_desc,activity_data[index][1]])
            except:
                output.append(["NA","NA","NA"])
                print(s)

        output = np.asarray(output)
        rank_index = np.argsort(np.asarray(output[:,-1],dtype=float))
        rank_output.append([output[:,0],output[:,1],rank_index])

    rank_output = np.asarray(rank_output)

    #spearman's R
    output_file = open(output_filename,"w+")
    output_file.write("Descriptor,Spearman R\n")
    for i in range(len(rank_list)):
        rank_dict = rank_list[i]
        activity_values = np.asarray(rank_output[:,1][i],dtype=float)
        rank_values = np.asarray(rank_output[:,2][i],dtype=float)
        output_file.write(rank_dict["desc_name"]+","+str(spearmanr(rank_values,activity_values)[0])+"\n")
    output_file.close()
