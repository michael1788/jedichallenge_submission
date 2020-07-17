import os, sys
import time
import argparse
import csv
from rdkit import Chem

sys.path.append('../')
from src import helper as hp
from src import fixed_parameters as fpf

parser = argparse.ArgumentParser(description='Extract desciptors representation')
parser.add_argument('-m','--mode', type=str, help='Mode. Choice: production or testing', required=True)
parser.add_argument('-aa','--acti_thres_active', type=int, help='Threshold in nm for the active (<=)', required=True)
parser.add_argument('-an','--acti_thres_inactive', type=int, help='Threshold in nm for inactive (>)', required=True)
parser.add_argument('-mp','--max_len_prot', type=int, help='Maximum length for proteins', required=True)
parser.add_argument('-ms','--max_len_smi', type=int, help='Maximum length for SMILES', required=True)
parser.add_argument('-d','--datapath', type=str, help='Path to the PDB database in .tsv', required=True)
parser.add_argument('-s','--savepath', type=str, help='Path to save the extracted data', required=True)
parser.add_argument('-kd','--with_kd', action='store_true', help='If present, kd measures considered', required=False)
parser.add_argument('-ki','--with_ki', action='store_true', help='If present, ki measures considered', required=False)
parser.add_argument('-ic50','--with_ic50', action='store_true', help='If present, ic50 measures considered', required=False)
parser.add_argument('-ec50','--with_ec50', action='store_true', help='If present, ec50 measures considered', required=False)
parser.add_argument('-v','--verbose', action='store_true', help='If present, verbosity included', required=False)
parser.set_defaults(feature=True)


def is_input_ok(input_, max_len, d_vocab, mode):
    """
    input_: smi (str) if mode is smi
    input_: protein (str) if mode is prot
    max_len: maximum length to pass the check
    d_vocab: smi vocab is mode is smi
    d_vocab: prot vocab if mode is prot
    mode: smi or prot (str)
    return: True if the input path the check,
    else False
    """
    if input_=='': 
        return False
    
    if mode=='prot': tokenized = list(input_)
    elif mode=='smi': tokenized = hp.smi_tokenizer(input_)
    else: raise ValueError('mode does not exist')
        
    for x in tokenized:
        if x not in d_vocab:
            return False
    if len(tokenized)>max_len:
        return False
    return True

def get_activity(d_acti):
    """
    Get activity with highest priority
    d_acti: dict with activity_name:value
    Return: the highest priority activity (str)
    """
    if d_acti['Kd']:
        return d_acti['Kd']
    elif d_acti['Ki']:
        return d_acti['Ki']
    elif d_acti['IC50']:
        return d_acti['IC50']
    elif d_acti['EC50']:
        return d_acti['EC50']
    else:
        return None


if __name__ == '__main__':
    
    start = time.time()
    
    ####################################
    # get back parameters
    args = vars(parser.parse_args())
    
    mode = args['mode']
    acti_thres_active = args['acti_thres_active']
    acti_thres_inactive = args['acti_thres_inactive']
    max_len_prot = args['max_len_prot']
    max_len_smi = args['max_len_smi']
    datapath = args['datapath']
    savepath = args['savepath']
    
    with_kd = args['with_kd']
    with_ki = args['with_ki']
    with_ic50 = args['with_ic50']
    with_ec50 = args['with_ec50']
    
    verbose = args['verbose']
    
    if mode not in ['testing', 'production']:
        raise ValueError('Your mode is not valid')
    ####################################


    
    
    ####################################
    # start extraction
    all_smi = []
    all_protein = []
    all_binary = []
    
    d_vocab_prot = fpf.p_token_indices
    d_vocab_smi = fpf.s_token_indices
    
    with open(datapath) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        for i,line in enumerate(tsvreader):
            if mode=='testing':
                # we stay in low data
                # regime to test quickly
                if i==1000:
                    break
            if verbose:
                if i%1000==0: print(f'current index: {i}')
            if i>0:
                smi = line[1]
                organism = line[7]
                n_chains = line[36]
                protein = line[37]

                # get the canon smi
                try:
                    smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True)
                except:
                    print('error with smi at index: ', i)
                    print('correspoding data entry not taken')
                    continue
                    
                if organism!='sequence' and n_chains=='1' and is_input_ok(smi, max_len_smi, d_vocab_smi, 'smi') \
                 and is_input_ok(protein, max_len_prot, d_vocab_prot, 'prot'):
                    # prioritize activity
                    Ki = line[8]
                    IC50 = line[9]
                    Kd = line[10]
                    EC50 = line[11]
                    
                    d_acti = {'Ki':None, 'IC50':None, 'Kd':None, 'EC50':None}
                    
                    if with_kd:
                        d_acti['Kd'] = Kd
                    if with_ki:
                        d_acti['Ki'] = Ki
                    if with_ic50:
                        d_acti['IC50'] = IC50
                    if with_ec50:
                        d_acti['EC50'] = EC50
                    
                    # prioritize activity selected
                    # when multiple activity types are 
                    # reported
                    acti = get_activity(d_acti)
                    
                    if acti is not None and 'e+' not in acti and acti!='':
                        if '<' in acti: 
                            acti = acti.replace('<', '')
                            acti = float(acti)
                            # here, if eg <150, we don't know if ok
                            # so we only take below <100 (ex acti_thres)
                            if acti<=acti_thres_active: 
                                binary_score=1
                                all_smi.append(smi)
                                all_protein.append(protein)
                                all_binary.append(binary_score)
                        elif '>' in acti: 
                            acti = acti.replace('>', '')
                            acti = float(acti)
                            # here, if eg >50, we don't know if ok
                            # so we only take >100 as negative example (ex acti_thres)
                            if acti>acti_thres_inactive: 
                                binary_score=0
                                all_smi.append(smi)
                                all_protein.append(protein)
                                all_binary.append(binary_score)
                        else:
                            acti = float(acti)
                            if acti<=acti_thres_active: 
                                binary_score=1
                            elif acti>acti_thres_inactive: 
                                binary_score=0
                                
                            all_smi.append(smi)
                            all_protein.append(protein)
                            all_binary.append(binary_score)
    if verbose:
        print('Data extracted - starting duplicates removal')
    ####################################                        
        
    
    
    
    
    ####################################
    # Deal with duplicates and save
    pairs_w_score = [(p,s,b) for p,s,b in zip(all_protein, all_smi, all_binary)]
    unique_pairs_w_score = list(set(pairs_w_score))
    
    if verbose:
        print('Starting the first pass to deal with duplicates')
    temp_all_clean = {}
    for j,x in enumerate(unique_pairs_w_score):
        if verbose:
            if j%10000==0: print(f'Current pair: {j}')
        
        pair = x[:2]
        score = x[2]
        
        if pair not in temp_all_clean:
            temp_all_clean[pair] = [score] 
        else:
            temp_all_clean[pair].append(score)
    
    
    print('Starting the second pass')
    all_clean = []
    n_removed = 0
    n_same_duplicate = 0
    for k,v in temp_all_clean.items():
        if len(v)==1:
            datapoint = (k[0], k[1], v[0])
            all_clean.append(datapoint)
        # case with duplicates,
        # but same binary activity
        elif len(set(v))==1:
            datapoint = (k[0], k[1], v[0])
            all_clean.append(datapoint)
            n_same_duplicate+=1
        else:
            n_removed+=1
            
    print(f'N remaining data points: {len(all_clean)}')
    print(f'Duplicates removed because of inconsistent binary activity: {n_removed}')
    print(f'Entries with multiple same binary activity, so taken once : {n_same_duplicate}')
          
    os.makedirs(savepath, exist_ok=True)
    hp.save_pkl(f'{savepath}all_clean.pkl', all_clean)
    ####################################
    
    
    end = time.time()
    print(f'BindingDB extraction DONE in {end - start:.04} seconds')