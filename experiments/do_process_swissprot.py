import os, sys
import time
import argparse

from Bio import SeqIO

sys.path.append('../../')
from src import helper as hp
from src import fixed_parameters as fpf

parser = argparse.ArgumentParser(description='Process Swiss-Prot data')
parser.add_argument('-d','--datapath', type=str, help='Path to the Swiss-Prot data', required=True)
parser.add_argument('-s','--savepath', type=str, help='Path to save the extracted data', required=True)
parser.add_argument('-m','--max_len', type=int, help='Proteins maximun length in number of amino acids', required=True)
parser.add_argument('-v','--verbose', action='store_true', help='If present, verbosity included', required=False)
parser.set_defaults(feature=True)

if __name__ == '__main__':
    
    start = time.time()
    
    ####################################
    # get back parameters
    args = vars(parser.parse_args())
    
    datapath = args['datapath']
    savepath = args['savepath']
    max_len = args['max_len']
    verbose = args['verbose']
    ####################################


    
    
    ####################################
    # start processing
    d_vocab = fpf.p_token_indices
    
    path = f'{datapath}uniprot_sprot.fasta'

    all_seq = []
    
    i=0
    for record in SeqIO.parse(path, "fasta"):
        if verbose:
            if i%10000==0: print(f'current data being processed: {i}')
        seq = str(record.seq)
        if len(seq)<=max_len:
            isok=True
            for x in seq:
                if x not in d_vocab:
                    isok=False
                    
            if isok:
                all_seq.append(seq)
        i+=1
      
    unique_prot = list(set(all_seq))
    if verbose:
        print(f'Unique protein saved: {len(unique_prot)}')
        
    hp.write_in_file(f'{savepath}all_prot.txt', unique_prot)
    
    end = time.time()
    print(f'Data prepration for Swiss-Prot DONE in {end - start:.04} seconds')
    ####################################
    
    