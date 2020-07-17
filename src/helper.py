import pickle
import pandas as pd
import re

def smi_tokenizer(smi):
    """
    Tokenize a SMILES
    """
    pattern =  "(\[|\]|Xe|Ba|Rb|Ra|Sr|Dy|Li|Kr|Bi|Mn|He|Am|Pu|Cm|Pm|Ne|Th|Ni|Pr|Fe|Lu|Pa|Fm|Tm|Tb|Er|Be|Al|Gd|Eu|te|As|Pt|Lr|Sm|Ca|La|Ti|Te|Ac|Si|Cf|Rf|Na|Cu|Au|Nd|Ag|Se|se|Zn|Mg|Br|Cl|U|V|K|C|B|H|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%\d{2}|\d)"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]

    return tokens


def load_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pkl(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        
def write_in_file(path, data):
    with open(path, 'w+') as f:
        for item in data:
            f.write("%s\n" % item)
            
def read_with_pd(path, delimiter='\t', header=None):
    data_pd = pd.read_csv(path, delimiter=delimiter, header=header)
    return data_pd[0].tolist() 


