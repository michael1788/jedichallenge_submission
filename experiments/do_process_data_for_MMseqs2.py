import os, sys
import time
import argparse

from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO

parser = argparse.ArgumentParser(description='Get data ready for MMseqs2')
parser.add_argument('-d','--datapath', type=str, help='Path to the PDB database in .tsv', required=True)
parser.add_argument('-s','--savepath', type=str, help='Path to save the extracted data', required=True)
parser.add_argument('-v','--verbose', action='store_true', help='If present, verbosity included', required=False)
parser.set_defaults(feature=True)

if __name__ == '__main__':
    
    start = time.time()
    
    ####################################
    # get back parameters
    args = vars(parser.parse_args())
    
    datapath = args['datapath']
    savepath = args['savepath']
    verbose = args['verbose']
    ####################################


    
    
    ####################################
    # start processing
    path = f'{datapath}/all_clean.pkl'
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    # get back the proteins
    data_prot = [x[0] for x in data]
    data_prot = list(set(data_prot))
    
    # create the records
    data_records = []
    data_id_to_seq = {}
        
    for i,x in enumerate(data_prot):
        record = SeqRecord(Seq(x), f'{i}', '', '')
        data_records.append(record)
        data_id_to_seq[f'{i}'] = x
        
    # save 
    SeqIO.write(data_records, f'{savepath}/data.fasta', 'fasta')
          
    end = time.time()
    print(f'Data prepration for MMseqs2 DONE in {end - start:.04} seconds')
    ####################################
    
    