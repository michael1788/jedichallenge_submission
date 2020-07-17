import os, sys
import time
import argparse
import warnings

from Bio import SeqIO

sys.path.append('../../')
from src import helper as hp
from src import fixed_parameters as fpf

parser = argparse.ArgumentParser(description='Create train-valid split for pretraining data')
parser.add_argument('-f','--filename', type=str, help='Filename to the data', required=True)
parser.add_argument('-s','--savepath', type=str, help='Path to save the extracted data', required=True)
parser.add_argument('-t','--t_split', type=float, help='Ratio of data to be used for training', required=True)
parser.add_argument('-v','--verbose', action='store_true', help='If present, verbosity included', required=False)
parser.set_defaults(feature=True)

if __name__ == '__main__':
    
    start = time.time()
    
    ####################################
    # get back parameters
    args = vars(parser.parse_args())
    
    datapath = args['filename']
    savepath = args['savepath']
    t_split = args['t_split']
    verbose = args['verbose']
    
    if t_split>1.0 or t_split<0:
        raise ValueError('You cannot have a ratio smaller than 0 or bigger than 1')
    if t_split<0.75:
        warnings.warn('It seems that you are using a very low amount of your data for traning')
    ####################################


    
    
    ####################################
    # start processing
    data = hp.read_with_pd(filename)
    
    #random split with 95% of the data in the training set
    all_idx = np.arange(len(data))
    np.random.shuffle(all_idx)
    cut = int(len(all_idx)*t_split)
    
    idx_train = all_idx[:cut]
    idx_val = all_idx[cut:]
    if verbose:
        print(f'N data used for training: {len(idx_train)}')
        print(f'N data used for validation: {len(idx_val)}')
    
    # create the partitions for the data generator
    partition = {}
    partition['train'] = idx_train
    partition['val'] = idx_val
    
    hp.save_pkl(f'{savepath}partitions.pkl', partition)
    
    end = time.time()
    print(f'Train-val split for Swiss-Prot DONE in {end - start:.04} seconds')
    ####################################
    
    