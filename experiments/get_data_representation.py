import os, sys
import time
import argparse
import glob
import numpy as np
import random
from rdkit.Chem import AllChem as Chem

import multiprocessing
from joblib import Parallel, delayed

sys.path.append('../')
from src import helper as hp
from src import helper_clm as hp_clm
from src import fixed_parameters as fpf

seed = 16
random.seed(seed)

parser = argparse.ArgumentParser(description='Extract desciptors representation')
parser.add_argument('-m','--mode', type=str, help='Mode of the run, either testing or production', required=True)
parser.add_argument('-r','--rpr', type=str, help='Representation mode to extract', required=True)
parser.add_argument('-d','--datapath', type=str, help='Path to the data', required=True)
parser.add_argument('-s','--savepath', type=str, help='Path to save the data', required=True)
parser.add_argument('-n','--nworkers', type=int, help='Number of pool workers', default=1, required=False)
parser.add_argument('-sl','--s_max_len', type=int, help='SMILES max len', default=None, required=False)
parser.add_argument('-pl','--p_max_len', type=int, help='Prot max len', default=None, required=False)
parser.add_argument('-v','--verbose', action='store_true', help='If present, verbosity included', required=False)
parser.set_defaults(feature=True)


def get_rep(sequence, model, tokenizer):
    """
    sequence: protein aa sequence (str)
    return: model representation (torch tensor)
    """
    token_ids = torch.tensor([tokenizer.encode(sequence)])
    output = model(token_ids)
    sequence_output = output[0]
    mean_output = torch.mean(sequence_output[0], dim=0)
    return mean_output

def smi_to_int(smi, d_vocab):
    """
    smi: a smi string
    d_vocab: dict with key int,
    value char in smi vocab
    return: the input smi tokenized
    """
    t_smi = hp.smi_tokenizer(smi)
    return [d_vocab[x] for x in t_smi]

def get_data_to_idx_mapping(all_data): 
    
    d_all_s = dict(enumerate(all_data))
    unique_data_to_idx = {}
    
    for key, value in d_all_s.items():
        if value not in unique_data_to_idx:
            unique_data_to_idx[value] = [key]
        else:
            unique_data_to_idx[value].append(key)
    
    return unique_data_to_idx

def get_MACCS(i, smi):
    if i%1000==0: print(f'current index: {i}')
    mol = Chem.MolFromSmiles(smi)
    try:
        fp = Chem.GetMACCSKeysFingerprint(mol)
        fp = np.array([int(x) for x in fp])
        np.save(f'{f_savepath}{i}.npy', fp)
    except:
        pass
    
    return (i, smi)

def get_PROTrepr(i, prot, model, tokenizer, f_savepath):
    if i%100==0: print(f'current index: {i}')
    p_rep = get_rep(prot, model, tokenizer)
    np_rep = p_rep.detach().numpy()
    np.save(f'{f_savepath}{i}.npy', np_rep)
    
    return (i, prot)

def get_tok_smi(i, smi, token_indices, max_len_model, f_savepath):
    if i%1000==0: print(f'current idx: {i}')
    
    pad_char = fpf.s_pad_char
    start_char = fpf.s_start_char
    end_char = fpf.s_end_char
    
    token_list = hp.smi_tokenizer(smi)
    token_list = [start_char] + token_list + [end_char]
    padding = [pad_char]*(max_len_model - len(token_list))
    token_list.extend(padding)
    token_list = np.array([token_indices[x] for x in token_list])
    
    np.save(f'{f_savepath}{i}.npy', token_list)
    
    return (i, smi)

def get_tok_prot(i, prot, token_indices, max_len_model, f_savepath):
    if i%1000==0: print(f'current idx: {i}')
    
    pad_char = fpf.p_pad_char
    start_char = fpf.p_start_char
    end_char = fpf.p_end_char
    
    token_list = list(prot)
    token_list = [start_char] + token_list + [end_char]
    padding = [pad_char]*(max_len_model - len(token_list))
    token_list.extend(padding)
    token_list = np.array([token_indices[x] for x in token_list])
    
    np.save(f'{f_savepath}{i}.npy', token_list)
    
    return (i, prot)

def do_z_norma(path, rep_dim):
    # this is necessary cause with, eg whales,
    # not all molecules can be projected, so
    # we have missing id, thus not the number
    # of unique prot from our whole set
    n_unique = len(glob.glob1(path,'*.npy'))
    all_data = np.zeros([n_unique, rep_dim])

    d_i_filename = {}
    i=0
    for filename in os.listdir(path):
        if filename.endswith('.npy'):
            all_data[i] = np.load(f'{path}{filename}')
            d_i_filename[i] = filename
            i+=1
                         
    mean = np.mean(all_data, axis=0)
    std = np.std(all_data, axis=0)
                         
    z_data = (all_data - mean)/std
    
    for i in range(n_unique):
        z_rep = z_data[i]
        filename = d_i_filename[i]
        np.save(f'{path}{filename}', z_rep)
        
    return mean, std
    
if __name__ == '__main__':
    
    ####################################
    # get back parameters
    args = vars(parser.parse_args())
    
    mode = args['mode']
    rpr = args['rpr']
    datapath = args['datapath']
    savepath = args['savepath']
    verbose = args['verbose']
    nworkers = args['nworkers']
    
    if mode not in ['testing', 'production']:
        raise ValueError('Your mode does not exist')
    if rpr not in ['protein', 'morgan', 'clm', 'binary', 'maccs', 'tok_smi', 'tok_prot']:
        raise ValueError('Your rpr does not exist')
    ####################################
    
    
    
    
    ####################################
    # get back data
    all_data = hp.load_pkl(f'{datapath}all_clean.pkl')
    
    all_protein, all_smi, all_binary = zip(*all_data)
    if mode=='testing':
        if rpr in ['unirep', 'protein']: small_n = 20
        else: small_n = 500
        all_protein = all_protein[:small_n]
        all_smi = all_smi[:small_n]
        all_binary = all_binary[:small_n]
    
    if verbose:
        print(f'data size: {len(all_protein)}')
    ####################################
    
    
    
    
    ####################################
    # project and save data
    start = time.time()
    
    f_savepath = f'{savepath}{rpr}/'
    os.makedirs(f_savepath, exist_ok=True)
    
    all_indices = {}
    
    if rpr=='protein':
        import torch
        from tape import ProteinBertModel, TAPETokenizer

        unique_protein = list(set(all_protein))
        print(f'n unique protein used to compute repr: {len(unique_protein)}')
        
        unique_prot_to_idx = get_data_to_idx_mapping(all_protein)
        
        # init protein pretrained model
        model = ProteinBertModel.from_pretrained('bert-base')
        tokenizer = TAPETokenizer(vocab='iupac') 
        
        results = Parallel(n_jobs=nworkers)(delayed(get_PROTrepr)(i,x,model,tokenizer,f_savepath) for i,x in enumerate(unique_protein))
        
        all_indices = {}
        for x in results:
            i = x[0]
            prot = x[1]
            _all_idx = unique_prot_to_idx[prot]
            for _idx in _all_idx:
                all_indices[_idx] = i
        
        hp.save_pkl(f'{savepath}all_indices_{rpr}.pkl', all_indices)
        
        z_norma = True
        if z_norma:
            rep_dim = 768
            mean, std = do_z_norma(f'{f_savepath}', rep_dim)
            hp.save_pkl(f'{savepath}z_norma_param_{rpr}.pkl', {'mean':mean, 'std':std})
        
    elif rpr=='clm':
        from src import helper_clm as hp_clm
        from keras.models import load_model

        max_len_model = 100 + 2
        pad_char = 'A'
        start_char ='G'
        end_char = 'E'
        indices_token = hp_clm.indices_token
        token_indices = hp_clm.token_indices
        n_chars = len(token_indices)
        oh = hp_clm.OneHotEncode(max_len_model, n_chars, indices_token, token_indices, 
                                 pad_char, start_char, end_char)

        model = hp_clm.ModelState(n_chars, max_len_model, {'lr': 0.001})
        path = '../dev/mm_dev/O01/_83.h5'
        pretrained_model = load_model(path)
        
        for i,layer in enumerate(pretrained_model.layers):
            weights = layer.get_weights()
            model.model.layers[i+1].set_weights(weights)
        
        unique_smi = list(set(all_smi))
        print(f'n unique smi used to compute CLM representation: {len(unique_smi)}')
        
        for i,smi in enumerate(unique_smi):
            if verbose:
                if i%1000==0: print(f'current index: {i}')
                    
            X = oh.one_smi_to_onehot(smi)
            output_clm = model.model.predict(X)
            pred_state_h = output_clm[1]            
            pred_h_torch = torch.FloatTensor(pred_state_h)
            
            indices = {idx:i for idx,x in enumerate(all_smi) if x==smi}
            all_indices.update(indices)
            
            torch.save(pred_h_torch, f'{f_savepath}{i}.pt')
        
        hp.save_pkl(f'{savepath}all_indices_{rpr}.pkl', all_indices)
        
    elif rpr=='morgan':
        unique_smi = list(set(all_smi))
        problematic_id = []
        print(f'n unique smi used to compute Morgan fingerprint: {len(unique_smi)}')
        
        for i,smi in enumerate(unique_smi):
            if verbose:
                if i%1000==0: print(f'current index: {i}')
            
            mol = Chem.MolFromSmiles(smi)
            try:
                fp = Chem.GetMorganFingerprintAsBitVect(mol, useChirality=True, radius=2, nBits=1024)
                fp = torch.FloatTensor(fp)
                                
                indices = {idx:i for idx,x in enumerate(all_smi) if x==smi}
                all_indices.update(indices)
                
                torch.save(fp, f'{f_savepath}{i}.pt')
            
            except:
                print(f'prob at {i} for morgan')
                problematic_id.append(i)
        
        hp.save_pkl(f'{savepath}all_indices_{rpr}.pkl', all_indices)
        hp.write_in_file(f'{savepath}problematic_id_morgan.txt', problematic_id)
        
    elif rpr=='maccs':
        from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint
        
        unique_smi = list(set(all_smi))
        print(f'n unique smi used to compute MACCS keys: {len(unique_smi)}')
        
        unique_smi_to_idx = get_data_to_idx_mapping(all_smi)
        
        results = Parallel(n_jobs=nworkers)(delayed(get_MACCS)(i,x) for i,x in enumerate(unique_smi))
        
        all_indices = {}
        for x in results:
            i = x[0]
            smi = x[1]
            _all_idx = unique_smi_to_idx[smi]
            for _idx in _all_idx:
                all_indices[_idx] = i
        
        hp.save_pkl(f'{savepath}all_indices_{rpr}.pkl', all_indices)
    
    elif rpr=='tok_smi':
        unique_smi = list(set(all_smi))
        print(f'n unique smi used to compute tok_smi: {len(unique_smi)}')
        
        unique_smi_to_idx = get_data_to_idx_mapping(all_smi)
        s_max_len = args['s_max_len']
        max_len_model = s_max_len+2
        token_indices = fpf.s_token_indices
        results = Parallel(n_jobs=nworkers)(delayed(get_tok_smi)(i,x,token_indices,max_len_model,f_savepath) for i,x in enumerate(unique_smi))
        
        all_indices = {}
        for x in results:
            i = x[0]
            smi = x[1]
            _all_idx = unique_smi_to_idx[smi]
            for _idx in _all_idx:
                all_indices[_idx] = i
        
        hp.save_pkl(f'{savepath}all_indices_{rpr}.pkl', all_indices)
        
    elif rpr=='tok_prot':
        unique_prot = list(set(all_protein))
        print(f'n unique prot used to compute tok_prot: {len(unique_prot)}')
        
        unique_prot_to_idx = get_data_to_idx_mapping(all_protein)
        p_max_len = args['p_max_len']
        max_len_model = p_max_len+2
        token_indices = fpf.p_token_indices
        results = Parallel(n_jobs=nworkers)(delayed(get_tok_prot)(i,x,token_indices,max_len_model,f_savepath) for i,x in enumerate(unique_prot))
        
        all_indices = {}
        for x in results:
            i = x[0]
            prot = x[1]
            _all_idx = unique_prot_to_idx[prot]
            for _idx in _all_idx:
                all_indices[_idx] = i
        
        hp.save_pkl(f'{savepath}all_indices_{rpr}.pkl', all_indices)
    
    elif rpr=='binary':
        d_all_binary = {}
        for i,binary in enumerate(all_binary):
            d_all_binary[i] = binary
        hp.save_pkl(f'{f_savepath}labels.pkl', d_all_binary)
    else:
        raise ValueError('rpr not valid')
    ####################################
    
        
    end = time.time()
    print(f'DESCRIPTORS extraction DONE in {end - start:.04} seconds')