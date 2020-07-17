import os, sys
import time
import argparse
import configparser
import numpy as np
import re
import ast

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import generator_clm as dg

sys.path.append('../')
from src import helper as hp

parser = argparse.ArgumentParser(description='Run CLM training')
parser.add_argument('-c','--configfile', type=str, help='Path to the config file', required=True)
parser.add_argument('-n','--ngpu', type=int, help='Number of GPUs', required=True)

class SeqModel():
    """Class to define the language model, i.e the neural net"""
    def __init__(self, n_chars, max_length, layers, dropouts, trainables, lr, batchnorm, verbose=False):  
        
        self.n_chars = n_chars
        self.max_length = max_length
        
        self.layers = layers
        self.dropouts = dropouts
        self.trainables = trainables
        self.lr = lr
        self.batchnorm = batchnorm
        
        self.model = None
        self.build_model()
        
    def build_model(self):
        
        self.model = tf.keras.Sequential()       
        i=0
        
        if self.batchnorm:
            self.model.add(tf.keras.layers.BatchNormalization(input_shape=(None, self.n_chars)))
        
            for neurons, dropout, trainable in zip(self.layers, self.dropouts, self.trainables):
                self.model.add(tf.keras.layers.LSTM(neurons, unit_forget_bias=True, dropout=dropout, 
                                           trainable=trainable, return_sequences=True))        
            self.model.add(tf.keras.layers.BatchNormalization()) 
        
        else:
            for neurons, dropout, trainable in zip(self.layers, self.dropouts, self.trainables):
                if i==0:
                    self.model.add(tf.keras.layers.LSTM(neurons, unit_forget_bias=True, dropout=dropout, 
                                               trainable=trainable, return_sequences=True, 
                                               input_shape=(None, self.n_chars)))
                else:
                    self.model.add(tf.keras.layers.LSTM(neurons, unit_forget_bias=True, dropout=dropout, 
                                               trainable=trainable, return_sequences=True))
                i+=1
        
        self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.n_chars, activation='softmax')))
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)
        
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        
def create_model_checkpoint(save_path):
    """ Function to save the trained model during training """
    filepath = save_path + 'best.h5' 
    checkpointer = ModelCheckpoint(filepath=filepath,
                                   verbose=0,
                                   save_best_only=True)

    return checkpointer
    
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

pad_char = 'A'
start_char = 'G'
end_char = 'E'


if __name__ == '__main__':
    
    start = time.time()
    
    ####################################
    # get back parameters
    args = vars(parser.parse_args())
    
    verbose = True
    configfile = args['configfile']
    ngpu = args['ngpu']
    config = configparser.ConfigParser()
    config.read(configfile)
    
    if ngpu>1:
        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    
    if verbose: print('\nSTART EXPERIMENT')
    ####################################
    
    
    
    
    
    ####################################
    # Path to save the checkpoints
    exp_name = configfile.split('/')[-1].replace('.ini','')
    savedir = str(config['DATA']['savedir'])
    save_path = f'{savedir}/{exp_name}/'
    os.makedirs(save_path, exist_ok=True)
    ####################################
    
    
    
    
    
    ####################################
    # Get parameters back
    # batch size depends on the number of GPUs
    # with tf strategy
    batch_size = int(config['MODEL']['batch_size'])
    batch_size = batch_size * ngpu
    
    num_workers = int(config['MODEL']['num_workers']) 
    max_len = int(config['MODEL']['max_len'])
    patience_lr = int(config['MODEL']['patience_lr'])
    min_lr = float(config['MODEL']['min_lr'])
    factor = float(config['MODEL']['factor'])
    epochs = int(config['MODEL']['epochs'])
    layers = ast.literal_eval(config['MODEL']['neurons'])
    dropouts = ast.literal_eval(config['MODEL']['dropouts'])
    trainables = ast.literal_eval(config['MODEL']['trainables'])
    lr = float(config['MODEL']['lr'])
    batchnorm = config.getboolean('MODEL', 'batchnorm')
    
    vocab_size = len(indices_token)
    max_len_model = max_len+2
    
    if verbose:
        print(f'batch_size: {batch_size}')
        print(f'n epochs: {epochs}')
    ####################################
    
    
                       
                       
                       
    ####################################
    # Define monitoring
    monitor = 'val_loss'
    lr_reduction = ReduceLROnPlateau(monitor=monitor, 
                                     patience=patience_lr, 
                                     verbose=0, 
                                     factor=factor, 
                                     min_lr=min_lr)
    checkpointer = create_model_checkpoint(save_path)
    
    early_stopper = EarlyStopping(monitor=monitor, patience=3)
    ####################################
                       
    
    
    
    
    ####################################
    # Path to the data
    datadir = str(config['DATA']['datadir'])
    
    # load partitions
    partition = {}
    path_partition_train = f'{datadir}/idx_tr.pkl'
    path_partition_valid = f'{datadir}/idx_val.pkl'
    
    partition['train'] = hp.load_pkl(path_partition_train)
    partition['val'] = hp.load_pkl(path_partition_valid)
    ####################################
    
    
    
                       
                       
    ####################################
    # Create the generators
    path_data = f'{datadir}/c27_canon_smi.txt'
    
    tr_generator = dg.DataGenerator(partition['train'],
                                    batch_size, 
                                    max_len_model,
                                    path_data,
                                    vocab_size,
                                    indices_token,
                                    token_indices,
                                    pad_char,
                                    start_char,
                                    end_char,
                                    shuffle=True)
    
    val_generator = dg.DataGenerator(partition['val'], 
                                     batch_size, 
                                     max_len_model, 
                                     path_data,
                                     vocab_size,
                                     indices_token,
                                     token_indices,
                                     pad_char,
                                     start_char,
                                     end_char,
                                     shuffle=True)
    ####################################

    
    
    
    
    ####################################
    # Create the model and train
    # Open a strategy scope.
    if ngpu>1:
        with strategy.scope():
            seqmodel = SeqModel(vocab_size, max_len_model, layers, dropouts, trainables, lr, batchnorm)
    else:
        seqmodel = SeqModel(vocab_size, max_len_model, layers, dropouts, trainables, lr, batchnorm)
        
    if config.getboolean('RESTART', 'restart'):
        # Load the pretrained model
        path_model = config['RESTART']['path_model']
        if path_model is None:
            raise ValueError('You did not provide a path to a model to be loaded for the restart')
        seqmodel.model = tf.keras.models.load_model(path_model)
        
        
    history = seqmodel.model.fit_generator(generator=tr_generator,
                                           validation_data=val_generator,
                                           use_multiprocessing=True,
                                           epochs=epochs,
                                           callbacks=[checkpointer,
                                                      lr_reduction,
                                                      early_stopper],
                                           workers=num_workers,
                                           verbose=2)
    
    
    hp.save_pkl(f'{save_path}history', history.history)
    end = time.time()
    print(f'TRAINING DONE in {end - start:.05} seconds')
    ####################################
