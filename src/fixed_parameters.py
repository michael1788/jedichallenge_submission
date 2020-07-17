# SMILES fixed parameters
s_pad_char = 'A'
s_start_char = 'G'
s_end_char = 'E'
s_indices_token = {0: s_pad_char,1: 'c',2: 'C',3: '(',4: ')',5: 'O',6: '1',7: '2',8: '=',9: 'N',10:'@',11: '[',12: ']',13: 'n',14: '3',15: 'H',16: 'F',17: '4',18: '-',19: 'S',20: 'Cl',21: '/',22: 's',23: 'o',24: '5',25: '+',26: '#',27: '\\',28: 'Br',29: 'P',30: '6',31: 'I',32: '7',33: s_end_char,34: s_start_char}
s_token_indices = {v: k for k, v in s_indices_token.items()}

# PROTEIN fixed parameters
p_pad_char = 'p'
p_start_char = 's'
p_end_char = 'e'
p_indices_token = {0: p_pad_char,1: 'C',2: 'D',3: 'E',4: 'F',5: 'G',6: 'H',7: 'I',8: 'K',9: 'L',10: 'M',11: 'N',12: 'P',13: 'Q',14: 'R',15: 'S',16: 'T',17: 'V',18: 'W',19: 'Y',20: 'A',21: p_end_char,22: p_start_char}
p_token_indices = {v: k for k, v in p_indices_token.items()}