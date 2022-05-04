'''Functions to construct Pandas dataframes from RDKit molecules'''

import numpy as np
from pandas import DataFrame, Series

def encode_factor(observations, levels):
    '''Given length n list of observations (strings) and length k list of
    levels (strings), output n by k matrix (pandas dataframe, with columns
    labeled by levels) of one-hot encodings'''
    factor_dict = dict((symbol, i) for i, symbol in enumerate(levels))
    factor_indices = [factor_dict[obs] for obs in observations]
    factor_onehot = np.identity(len(levels))[factor_indices,:]
    factor_df = DataFrame(factor_onehot, columns = levels, dtype = int)
    return factor_df

def extract_double(key, objs):
    '''Given an iterable sequence "obs" of atoms, bonds, or molecules (anything
    with a GetProp method), return the property specified by "key" as a Pandas
    series, with the name specified by "key". For a numeric property'''
    return Series([obj.GetDoubleProp(key) for obj in objs], name = key, dtype = np.float32)
