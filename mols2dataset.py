import numpy as np

def encode_factor(observations, levels):
    '''Given length n list of observations (strings) and length k list of
    levels (strings), output n by k matrix (numpy array) of one-hot
    encodings'''
    factor_dict = dict((symbol, i) for i, symbol in enumerate(levels))
    factor_indices = [factor_dict[obs] for obs in observations]
    factor_onehot = np.identity(len(levels))[factor_indices,:]
    return factor_onehot
