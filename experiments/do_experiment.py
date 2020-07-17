import os, sys
import time
import argparse
import ast
import configparser
import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import generator_fuego as data_generator

sys.path.append('../')
from src import helper as hp
from src import fixed_parameters as fpf

parser = argparse.ArgumentParser(description='Run experiment')
parser.add_argument('-c','--configfile', type=str, help='Path to the config file', required=True)
parser.add_argument('-cv','--cv_fold', type=int, help='Cross-validation fold number', required=True)
parser.add_argument('-r','--repeat', type=int, help='Id of the repeat', required=False)


class FullModel():
    """Class to define the language model, i.e the neural net"""
    def __init__(self, n_p_chars, p_layers, p_dropouts, p_trainables,
                       n_s_chars, s_layers, s_dropouts, s_trainables,
                       lr):
        
        self.n_p_chars = n_p_chars
        self.p_layers = p_layers
        self.p_dropouts = p_dropouts
        self.p_trainables = p_trainables
        self.n_s_chars = n_s_chars
        self.s_layers = s_layers
        self.s_dropouts = s_dropouts
        self.s_trainables = s_trainables
        self.lr = lr
        self.model = self.build_model()

    def build_model(self):
        ##################
        # PROTEIN input
        p_n_layers = len(self.p_layers)
        p_n = 0
        protein_input = tf.keras.layers.Input(shape=(None, self.n_p_chars), name="plm")
        in_ = tf.keras.layers.BatchNormalization()(protein_input)
        
        #protein_input = tf.keras.layers.BatchNormalization(input_shape=(None, self.n_p_chars), name="plm")
        for neurons, dropout, trainable in zip(self.p_layers, self.p_dropouts, self.p_trainables):
            if p_n==p_n_layers-1: p_rseq = False
            else: p_rseq = True
            if p_n==0: in_ = in_
            else: in_ = x_p
            x_p = tf.keras.layers.LSTM(neurons, unit_forget_bias=True, dropout=dropout, 
                              trainable=trainable, return_sequences=p_rseq)(in_)
            p_n+=1
        #x_p = tf.keras.layers.BatchNormalization()(x_p)
        ##################
        
        
        ##################
        # SMILES input
        s_n_layers = len(self.s_layers)
        s_n = 0
        smiles_input = tf.keras.layers.Input(shape=(None, self.n_s_chars), name="clm")
        in_ = tf.keras.layers.BatchNormalization()(smiles_input)
        
        #smiles_input = tf.keras.layers.BatchNormalization(input_shape=(None, self.n_s_chars), name="clm")
        for neurons, dropout, trainable in zip(self.s_layers, self.s_dropouts, self.s_trainables):
            if s_n==s_n_layers-1: s_rseq = False
            else: s_rseq = True
            if s_n==0: in_ = in_
            else: in_ = x_s
            x_s = tf.keras.layers.LSTM(neurons, unit_forget_bias=True, dropout=dropout, 
                              trainable=trainable, return_sequences=s_rseq)(in_)
            s_n+=1
        #x_s = tf.keras.layers.BatchNormalization()(x_s)
        ##################
        
        
        x = tf.keras.layers.concatenate([x_p, x_s])
        activity_pred = tf.keras.layers.Dense(1, name="activity")(x)
        
        # Instantiate an end-to-end model predicting both priority and department
        model = tf.keras.Model(
            inputs=[protein_input, smiles_input],
            outputs=activity_pred)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)
        
        model.compile(
            optimizer=optimizer,
            loss={"activity": tf.keras.losses.BinaryCrossentropy(from_logits=True)})
        
        return model
        
def create_model_checkpoint(period, save_path):
    """ Function to save the trained model during training """
    filepath = save_path + '{epoch:02d}.h5'
    checkpointer = ModelCheckpoint(filepath=filepath,
                                   verbose=0,
                                   save_best_only=False,
                                   period=period)
    return checkpointer


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
    configfile = args['configfile']
    cv_fold = args['cv_fold']
    repeat = args['repeat']
    config = configparser.ConfigParser()
    config.read(configfile)
    
    if verbose: print('\nSTART EXPERIMENT')
    ####################################
    
    
    
    
    
    ####################################
    # Path to save the checkpoints
    mode = str(config['EXPERIMENTS']['mode'])
    dir_exp = str(config['DATA']['savedir'])
    
    exp_name = configfile.split('/')[-1].replace('.ini','')
    if repeat:
        save_path = f'{dir_exp}{mode}/{exp_name}/{repeat}/{cv_fold}/'
    else:
        save_path = f'{dir_exp}{mode}/{exp_name}/{cv_fold}/'
    os.makedirs(save_path, exist_ok=True)
    ####################################
    
    
    
    
    
    ####################################
    # get parameters back 
    dataroot = str(config['DATA']['datadir'])
    
    num_workers = int(config['MODEL']['num_workers']) 
    batch_size = int(config['MODEL']['batch_size'])
    patience = int(config['MODEL']['patience'])
    shuffle = config.getboolean('MODEL', 'shuffle')
    save_model = config.getboolean('MODEL', 'save_model')
    epochs = int(config['MODEL']['epochs'])
    lr = float(config['MODEL']['lr'])
    
    if verbose:
        print(f'cv fold: {cv_fold}')
        print(f'batch_size: {batch_size}')
        print(f'n epochs: {epochs}')
        
    # Datasets
    if cv_fold==42:
        partition = hp.load_pkl(f'{dataroot}CV_partition.pkl')
        labels = hp.load_pkl(f'{dataroot}labels.pkl')
        all_indices_protein = hp.load_pkl(f'{dataroot}all_indices_tok_prot.pkl')
        all_indices_mol = hp.load_pkl(f'{dataroot}all_indices_tok_smi.pkl')
    else:
        if cv_fold==16: 
            pathcvfold = 0
        else:
            pathcvfold = cv_fold
        partition = hp.load_pkl(f'{dataroot}CF_fold/CV_partition_{pathcvfold}.pkl')
        labels = hp.load_pkl(f'{dataroot}labels/labels.pkl')
        all_indices_protein = hp.load_pkl(f'{dataroot}all_indices_tok_prot.pkl')
        all_indices_mol = hp.load_pkl(f'{dataroot}all_indices_tok_smi.pkl')
    ####################################
    
    
                       
                       
                       
    ####################################
    # Define monitoring
    if cv_fold==16:
        monitor = 'loss'
    else:
        monitor = 'val_loss'
    patience_lr = 3
    min_lr = 0.0005
    factor = 0.5
    lr_reduction = ReduceLROnPlateau(monitor=monitor, 
                                     patience=patience_lr, 
                                     verbose=0, 
                                     factor=factor, 
                                     min_lr=min_lr)
    
    period = 1
    checkpointer = create_model_checkpoint(period, save_path)    
    ####################################
                       
    
    
                       
                       
    ####################################
    # Create the generators
    p_input_dim = int(config['DATA']['p_input_dim']) 
    p_input_dim = p_input_dim + 2
    s_input_dim = int(config['DATA']['s_input_dim']) 
    s_input_dim = s_input_dim + 2
    
    if cv_fold==42:
        tr_generator = data_generator.DataGenerator(partition['train'],
                                                    batch_size, 
                                                    dataroot,
                                                    len(p_indices_token),
                                                    len(s_indices_token),
                                                    p_input_dim,
                                                    s_input_dim,
                                                    all_indices_protein,
                                                    all_indices_mol,
                                                    labels,
                                                    shuffle=shuffle)
    elif cv_fold==16:
        # training on all the data
        full_partition = partition['train'] + partition['validation']
        
        tr_generator = data_generator.DataGenerator(full_partition,
                                                    batch_size, 
                                                    dataroot,
                                                    len(p_indices_token),
                                                    len(s_indices_token),
                                                    p_input_dim,
                                                    s_input_dim,
                                                    all_indices_protein,
                                                    all_indices_mol,
                                                    labels,
                                                    shuffle=shuffle)
        
    else:
        tr_generator = data_generator.DataGenerator(partition['train'],
                                                    batch_size, 
                                                    dataroot,
                                                    len(p_indices_token),
                                                    len(s_indices_token),
                                                    p_input_dim,
                                                    s_input_dim,
                                                    all_indices_protein,
                                                    all_indices_mol,
                                                    labels,
                                                    shuffle=shuffle)
        
        val_generator = data_generator.DataGenerator(partition['validation'], 
                                                     batch_size, 
                                                     dataroot,
                                                     len(p_indices_token),
                                                     len(s_indices_token),
                                                     p_input_dim,
                                                     s_input_dim,
                                                     all_indices_protein,
                                                     all_indices_mol,
                                                     labels,
                                                     shuffle=shuffle)
    ####################################

    
    
    
    
    ####################################
    # Create model and Training loop
    # protein
    p_n_chars = len(p_indices_token)
    p_layers = ast.literal_eval(config['PMODEL']['neurons'])
    p_dropouts = ast.literal_eval(config['PMODEL']['dropouts'])
    p_trainables = ast.literal_eval(config['PMODEL']['trainables'])
    # smiles
    s_n_chars = len(s_indices_token)
    s_layers = ast.literal_eval(config['SMODEL']['neurons'])
    s_dropouts = ast.literal_eval(config['SMODEL']['dropouts'])
    s_trainables = ast.literal_eval(config['SMODEL']['trainables'])
    
    fuego = FullModel(p_n_chars, p_layers, p_dropouts, p_trainables,
                      s_n_chars, s_layers, s_dropouts, s_trainables,
                      lr)
    
    callbacks = [lr_reduction]
    if save_model:
        callbacks.append(checkpointer)
        
    # full model if fine-tuning experiment
    path_full_pretrained_model = config['EXPERIMENTS']['path_full_pretrained_model']
    if path_full_pretrained_model:
        print('Full pretrained model loaded')
        full_pretrained_model = tf.keras.models.load_model(path_full_pretrained_model)
        fuego.model = full_pretrained_model
    else:
        # load pretrained model if available
        # protein
        prot_path_weights = config['PMODEL']['path_weights']
        if prot_path_weights:
            pre_model = tf.keras.models.load_model(prot_path_weights)
            pre_weights = pre_model.get_weights()
    
            # batchnorm
            fuego.model.layers[2].set_weights(pre_weights[0:4])
            # lstm
            fuego.model.layers[4].set_weights(pre_weights[4:7])
            print('weights for proteins loaded')
            
        # smiles
        smi_path_weights = config['SMODEL']['path_weights']
        if smi_path_weights:
            pre_model = tf.keras.models.load_model(smi_path_weights)
            pre_weights = pre_model.get_weights()
    
            # batchnorm
            fuego.model.layers[3].set_weights(pre_weights[0:4])
            # lstm
            fuego.model.layers[5].set_weights(pre_weights[4:7])
            print('weights for smiles loaded')
        
            
    if cv_fold in [16, 42]:
        history = fuego.model.fit_generator(generator=tr_generator,
                                            use_multiprocessing=True,
                                            epochs=epochs,
                                            callbacks=callbacks,
                                            workers=num_workers)  
    else:
        history = fuego.model.fit_generator(generator=tr_generator,
                                            validation_data=val_generator,
                                            use_multiprocessing=True,
                                            epochs=epochs,
                                            callbacks=callbacks,
                                            workers=num_workers)
    
    # Save the loss history
    hp.save_pkl(f'{save_path}history', history.history)
    
    if cv_fold==16:
        fuego.model.save(f'{save_path}last.h5')
    
    
    end = time.time()
    print(f'EXPERIMENT DONE in {end - start:.05} seconds')
    ####################################