import os, sys
import time
import argparse
import csv

sys.path.append('../')
from src import helper as hp

parser = argparse.ArgumentParser(description='Processed data output from MMseqs2')
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
    with open(f'{datapath}/clusterRes_cluster.tsv') as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        for i,line in enumerate(tsvreader):
            cluster_repr = line[0]
            in_cluster = line[1]
            
            if cluster_repr in all_clusters_id:
                all_clusters_id[cluster_repr].append(in_cluster)
            else:
                all_clusters_id[cluster_repr] = [in_cluster]
           
    hp.save_pkl(f'{savepath}/data_id_to_seq.pkl', data_id_to_seq)
    hp.save_pkl(f'{savepath}/all_clusters_id.pkl', all_clusters_id)

    end = time.time()
    print(f'Data process from MMseqs2 DONE in {end - start:.04} seconds')
    ####################################
    
    