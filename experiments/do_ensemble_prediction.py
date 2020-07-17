import os, sys
import time
import argparse
import numpy as np

import tensorflow as tf

sys.path.append('../')
from src import helper as hp
from src import fixed_parameters as fpf

parser = argparse.ArgumentParser(description='Run experiment')
parser.add_argument('-i','--chunk_id', type=int, help='Id of the current data chunk', required=True)
parser.add_argument('-n','--n_chunk', type=int, help='Total number of chunks', required=True)
parser.add_argument('-r','--root', type=str, help='Root for models and data', required=True)
parser.add_argument('-m','--mode', type=str, help='If test, will run on a small data to test. Else choose deploy', required=True)


def do_chunkify(lst,n):
    """get n chunk of equal length except last to cover lst"""
    return [lst[i::n] for i in range(n)]

def one_hot_encode(tokenized_input, n_chars):
    output = np.zeros([len(tokenized_input), n_chars])
    for j, token in enumerate(tokenized_input):
        output[j, int(token)] = 1
    return output

def get_tok_smi(smi, token_indices, max_len_model, start_char, end_char, pad_char):
    
    token_list = hp.smi_tokenizer(smi)
    token_list = [start_char] + token_list + [end_char]
    padding = [pad_char]*(max_len_model - len(token_list))
    token_list.extend(padding)
    token_list = np.array([token_indices[x] for x in token_list])
    
    return token_list
    
def get_tok_prot(prot, token_indices, max_len_model, start_char, end_char, pad_char):
    
    token_list = list(prot)
    token_list = [start_char] + token_list + [end_char]
    padding = [pad_char]*(max_len_model - len(token_list))
    token_list.extend(padding)
    token_list = np.array([token_indices[x] for x in token_list])
    
    return token_list

# SMILES fixed parameters
s_pad_char = fpf.s_pad_char
s_start_char = fpf.s_start_char
s_end_char = fpf.s_end_char
s_indices_token = fpf.s_indices_token
s_token_indices = fpf.s_token_indices

# PROTEIN fixed parameters
p_pad_char = fpf.p_pad_char
p_start_char = fpf.p_start_char
p_end_char = fpf.p_end_char
p_indices_token = fpf.p_indices_token
p_token_indices = fpf.p_token_indices


if __name__ == '__main__':
    
    start = time.time()
    
    ####################################
    # get back parameters
    args = vars(parser.parse_args())
    
    verbose = True
    chunk_id = args['chunk_id']
    n_chunk = args['n_chunk']
    root = args['root']
    mode = args['mode']
    
    if mode not in ['test', 'deploy']:
        raise ValueError('Your mode does not exist. Pick: test or deploy')
    
    if mode=='test':
        savepath = f'{root}results/'
    else:
        savepath = f'{root}results/test/'
    os.makedirs(savepath, exist_ok=True)
    
    if verbose: print('\nSTART PREDICTIONS')
    ####################################
    
    
    
    
    
    ####################################
    if mode=='test':
        all_smi = hp.read_with_pd(f'{root}data_clean/small.txt')
    elif mode=='deploy':
        all_smi = hp.read_with_pd(f'{root}data_clean/all_smi_clean.txt')
        
    if n_chunk>1:
        data_chunkified = do_chunkify(all_smi, n_chunk)
        
        # -1 because the arg to the bash script will go,
        # eg for 10, from 1 to 10 included
        # so we want idx 0 to 9
        chunk_id = chunk_id-1
        all_smi = data_chunkified[chunk_id]
        print(f'N of data in the current chunk: {len(all_smi)}')
        
    n_data = len(all_smi)
    max_len_model = 100+2
    p_n_chars = len(p_token_indices)
    s_n_chars = len(s_token_indices)
    
    s_data = np.empty([n_data, max_len_model])

    for i,s in enumerate(all_smi):
        s_data[i] = get_tok_smi(s, s_token_indices, max_len_model, s_start_char, s_end_char, s_pad_char)
    
    
    # protein
    protein = hp.read_with_pd(f'{root}data_clean/rcsb_pdb_6LZE.txt')
    p_max_len_model = 1000+2
    p_data = np.empty([1, p_max_len_model])
    
    p_data[0] = get_tok_prot(protein, p_token_indices, p_max_len_model, p_start_char, 
                             p_end_char, p_pad_char)
    ####################################
    
    
    
    
    
    ####################################
    # start prediction
    n_model = 5
    all_y_preds = {}
    
    # we have only one prot
    p = p_data[0]
    p = one_hot_encode(list(p), p_n_chars)
    p = np.expand_dims(p, axis=0)
    
    for i in range(1,n_model+1):
        print(f'\ncurrent model: {i}')
        
        path = f'{root}A02_5repeats/{i}/16/last.h5'
        model = tf.keras.models.load_model(path)
        
        l_start = time.time()
        log_every = 200
        for n in range(n_data):
            print('current data ', n)
            if n%log_every==0 and n!=0: 
                l_end = time.time()
                l_time = l_end - l_start
                print(f'current data: {n}')
                print(f'{log_every} data done in: {l_time:.05} seconds')
                l_start = time.time()
                
            s = s_data[n]
            s = one_hot_encode(list(s), s_n_chars)
            s = np.expand_dims(s, axis=0)
            
            logit = model.predict([p,s])
            pred = tf.nn.sigmoid(logit)
            fpred = float(pred.numpy())
            
            if all_smi[n] in all_y_preds:
                all_y_preds[all_smi[n]].append(fpred)
            else:
                all_y_preds[all_smi[n]] = [fpred]
    
        hp.save_pkl(f'{savepath}chunk_{chunk_id}_model_{i}_all_y_preds.pkl', all_y_preds)
        
    end = time.time()
    print(f'PREDICTIONS DONE in {end - start:.05} seconds')
    ####################################