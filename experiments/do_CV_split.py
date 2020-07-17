import os, sys
import time
import argparse
import random
from sklearn.model_selection import KFold

sys.path.append('../')
from src import helper as hp

seed = 16
random.seed(seed)

parser = argparse.ArgumentParser(description='Create cross-validation folds')
parser.add_argument('-n','--n_fold', type=int, help='Number of cross-validation fold', required=True)
parser.add_argument('-d','--datapath', type=str, help='Path to the data', required=True)
parser.add_argument('-s','--savepath', type=str, help='Path to save the data', required=True)
parser.add_argument('-v','--verbose', action='store_true', help='If present, verbosity included', required=False)
parser.set_defaults(feature=True)

if __name__ == '__main__':
    
    start = time.time()
    
    ####################################
    # get back parameters
    args = vars(parser.parse_args())
    
    n_fold = args['n_fold']
    datapath = args['datapath']
    savepath = args['savepath']
    verbose = args['verbose']
    
    savepath = f'{savepath}CF_fold/'
    os.makedirs(savepath, exist_ok=True)
    ####################################

    
    
    
    ####################################
    # get the data
    data_id_to_seq = hp.load_pkl(f'{datapath}MMseqs2/data_id_to_seq.pkl')
    all_clusters_id = hp.load_pkl(f'{datapath}MMseqs2/all_clusters_id.pkl')
    cluster_ids = list(all_clusters_id.keys())
    all_ids = []
    for k,v in all_clusters_id.items():
        all_ids.extend(v)
    
    all_data = hp.load_pkl(f'{datapath}all_clean.pkl')
    all_protein, all_smi, all_binary = zip(*all_data)
    ####################################
    
    
    
    
    
    ####################################
    # Create the CV fold
    kf = KFold(n_splits=n_fold, random_state=seed, shuffle=True)
    
    # We do the CV folds on the cluster ids
    i=0
    for train_index, validation_index in kf.split(cluster_ids):
        if verbose:
            print(f'Current CV fold being processed: {i}')
        assert len(validation_index)+len(train_index)==len(cluster_ids)
        
        train_cluster_idx = [x for i,x in enumerate(cluster_ids) if i in train_index]
        val_cluster_idx = [x for i,x in enumerate(cluster_ids) if i in validation_index]
        
        data_id_tr = []
        data_id_val = []
        
        # get all the protein index from the cluster id
        tr_all_id = []
        for id_ in train_cluster_idx:
            # cluster_ids[id_] is a list of all
            # id belonging the the cluster id_,
            # including id_
            tr_all_id.extend(all_clusters_id[str(id_)])
        
        val_all_id = []
        for id_ in val_cluster_idx:
            val_all_id.extend(all_clusters_id[str(id_)])
            
        assert len(tr_all_id)+len(val_all_id)==len(all_ids)
        
        # get all the data id from the protein index
        for id_ in tr_all_id:
            seq = data_id_to_seq[id_]
            indices = [idx for idx,x in enumerate(all_protein) if x==seq]
            data_id_tr.extend(indices)  
        
        for id_ in val_all_id:
            seq = data_id_to_seq[id_]
            indices = [idx for idx,x in enumerate(all_protein) if x==seq]
            data_id_val.extend(indices)
        
        assert len(data_id_tr)+len(data_id_val)==len(all_protein)
        
        partition = {}
        partition['train'] = data_id_tr
        partition['validation'] = data_id_val
    
        
        hp.save_pkl(f'{savepath}/CV_partition_{i}.pkl', partition)
        i+=1
    ####################################
    
        
    end = time.time()
    print(f'CV folds DONE in {end - start:.04} seconds')