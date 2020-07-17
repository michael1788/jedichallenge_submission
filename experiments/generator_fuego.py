import re
import numpy as np
from tensorflow.keras.utils import Sequence
    
class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size, path_data, p_n_chars, s_n_chars,
                 p_input_dim, s_input_dim, ID_to_idx_protein, ID_to_idx_mol, labels, shuffle=True):
        'Initialization'
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.path_data = path_data
        self.p_n_chars = p_n_chars
        self.s_n_chars = s_n_chars
        self.p_input_dim = p_input_dim
        self.s_input_dim = s_input_dim
        self.ID_to_idx_protein = ID_to_idx_protein
        self.ID_to_idx_mol = ID_to_idx_mol
        self.labels = labels
        self.shuffle = shuffle       
        
        self.on_epoch_end()
        
    def one_hot_encode(self, tokenized_input, n_chars):
        output = np.zeros([len(tokenized_input), n_chars])
        for j, token in enumerate(tokenized_input):
            output[j, token] = 1
        return output
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        Xp, Xs, y = self.__data_generation(list_IDs_temp)

        return (Xp, Xs), y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, list_IDs_temp):
        'Generates batch of data containing batch_size samples' 
        
        Input_p = np.empty((self.batch_size, self.p_input_dim, self.p_n_chars))
        Input_s = np.empty((self.batch_size, self.s_input_dim, self.s_n_chars))
        y = np.empty((self.batch_size, 1))
        
        for i, ID in enumerate(list_IDs_temp):
            # get inputs
            idx_prot = self.ID_to_idx_protein[ID]      
            tok_protein = np.load(f'{self.path_data}tok_prot/{idx_prot}.npy')
            X_p = self.one_hot_encode(tok_protein, self.p_n_chars)
            idx_mol = self.ID_to_idx_mol[ID]
            tok_smiles = np.load(f'{self.path_data}tok_smi/{idx_mol}.npy')
            X_s = self.one_hot_encode(tok_smiles, self.s_n_chars)
            
            Input_p[i] = X_p
            Input_s[i] = X_s
            
            # get label
            y[i] = self.labels[ID]
           
        return Input_p, Input_s, y