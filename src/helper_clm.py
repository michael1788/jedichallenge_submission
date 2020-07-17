from keras import backend as K
from keras.layers import Dense, LSTM, BatchNormalization, Bidirectional, Embedding, Flatten, Dropout, Input
from keras.layers import Activation, TimeDistributed
from keras.optimizers import Adam
from keras.models import Sequential, load_model, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import keras

import numpy as np


indices_token = {0: 'A',
                     1: 'c',
                     2: 'C',
                     3: '(',
                     4: ')',
                     5: 'O',
                     6: '1',
                     7: '2',
                     8: '=',
                     9: 'N',
                     10:'@',
                     11: '[',
                     12: ']',
                     13: 'n',
                     14: '3',
                     15: 'H',
                     16: 'F',
                     17: '4',
                     18: '-',
                     19: 'S',
                     20: 'Cl',
                     21: '/',
                     22: 's',
                     23: 'o',
                     24: '5',
                     25: '+',
                     26: '#',
                     27: '\\',
                     28: 'Br',
                     29: 'P',
                     30: '6',
                     31: 'I',
                     32: '7',
                     33: 'E',
                     34: 'G'}
token_indices = {v: k for k, v in indices_token.items()}


class ModelState():
    def __init__(self, n_chars, max_length, param):  
        self.n_chars = n_chars
        self.max_length = max_length
        self.lr = param['lr']
        
        self.model = None
        self.build_model()
        
    def build_model(self):
        input_ = Input(shape=(self.max_length, self.n_chars))
        x = BatchNormalization()(input_)
        x, state_h, state_c = LSTM(1024, unit_forget_bias=True, dropout=0.4, 
                                   trainable=True, return_sequences=True, return_state=True)(x)
        x = LSTM(256, unit_forget_bias=True, dropout=0.4, 
                trainable=True, return_sequences=True)(x)
        x = BatchNormalization()(x)
        output_ = TimeDistributed(Dense(self.n_chars, activation='softmax'))(x)
                        
        self.model = Model(inputs=input_, outputs=[output_, state_h])
                            
        optimizer = Adam(lr=self.lr)
        self.model.compile(loss='mean_squared_error', 
                           optimizer=optimizer)
        
class OneHotEncode():
    def __init__(self, max_len_model, n_chars, indices_token, token_indices, pad_char, start_char, end_char):
        'Initialization'
        self.max_len_model = max_len_model
        self.n_chars = n_chars
        
        self.pad_char = pad_char
        self.start_char = start_char
        self.end_char = end_char

        self.indices_token = indices_token
        self.token_indices = token_indices

    def one_hot_encode(self, token_list, n_chars):
        output = np.zeros((token_list.shape[0], n_chars))
        for j, token in enumerate(token_list):
            output[j, token] = 1
        return output
    
    def smi_to_int(self, smi):
        """
        this will turn a list of smiles in string format
        and turn them into a np array of int, with padding
        """
        token_list = smi_tokenizer(smi)
        token_list = [self.start_char] + token_list + [self.end_char]
        padding = [self.pad_char]*(self.max_len_model - len(token_list))
        token_list.extend(padding)
        int_list = [self.token_indices[x] for x in token_list]
        return np.asarray(int_list)
    
    def int_to_smile(self, array):
        """ 
        From an array of int, return a list of 
        molecules in string smile format
        Note: remove the padding char
        """
        all_smi = []
        for seq in array:
            new_mol = [self.indices_token[int(x)] for x in seq]
            all_smi.append(''.join(new_mol).replace(self.pad_char, ''))
        return all_smi

    def clean_smile(self, smi):
        """ remove return line symbols """
        smi = smi.replace('\n', '')
        return smi
    
    def smile_to_onehot(self, path_data):
        
        f = open(path_data)
        lines = f.readlines()
        n_data = len(lines)
        
        X = np.empty((n_data, self.max_len_model, self.n_chars), dtype=int)
        
        for i,smi in enumerate(lines):
            # remove return line symbols
            smi = self.clean_smile(smi)
            # tokenize
            int_smi = self.smi_to_int(smi)
            # one hot encode
            X[i] = self.one_hot_encode(int_smi, self.n_chars)
            
        return X
    
    def one_smi_to_onehot(self, smi):
        
        X = np.empty((1, self.max_len_model, self.n_chars), dtype=int)
        # remove return line symbols
        smi = self.clean_smile(smi)
        # tokenize
        int_smi = self.smi_to_int(smi)
        # one hot encode
        X[0] = self.one_hot_encode(int_smi, self.n_chars)
            
        return X