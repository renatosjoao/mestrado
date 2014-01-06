##########################################
#Renato Stoffalette Joao
#
##########################################

__author__ = "Renato Stoffalette Joao(renatosjoao@gmail.com)"
__version__ = "$Revision: 0.1 $"
__date__ = "$Date: 2013// $"
__copyright__ = "Copyright (c) 2013 Renato SJ"
__license__ = "Python"


import numpy as np
import xplutil as xplutil

def read_from_XPL(xpl_file_path):
    """xplutil.py already reads from XPL file.

    Parameters
    ----------
    xpl_file_path : '/home/jdoe/path_to_xpl'
                    The path to the XPL file.

    Returns
    -------
    result : ExampleData(data, freq0, freq1, winshape, windata, filename)
            Same as xplutil returns.

    """
    result = xplutil.read_xpl(xpl_file_path)
    return result

def error(w0, w1):
    """ This is a function to calculate the error for the current interaction

    Parameters
    ----------
    w0 : array-like of shape = [n, 1]
        Label 0 frequency table.

    w1 : array-like of shape = [n, 1]
        Label 1 frequency table.

    Returns
    -------
    epsilon_t : double
        The error value for the current iteration.

    """
    epsilon_t = np.sum(np.min((w0, w1), axis=0))
    return epsilon_t

def beta_factor(epsilon_t):
    """ This is a function to calculate the beta_t factor

    Parameters
    ----------
    epsilon_t : double
        Error for the current iteraction.

    Returns
    -------
     beta_t : double
         Beta value for the current iteration

     """
    beta_t = epsilon_t / (1.0 - epsilon_t)
    return beta_t


def make_decision(w0, w1):
    """ This is a utility function to make a decision for each pattern
    based on w0 (weight for label 0 ) and w1(weight for label 1).It takes as
    input the tables with w0 and w1 frequencies, compare those values and make a decision.

    Parameters
    ----------
    w0 : array-like of shape = [n, 1]
        Label 0 frequency table.

    w1 : array-like of shape = [n, 1]
        Label 1 frequency table.

    Returns
    -------
    decision_table :  array-like of shape = [n, 1]
        This is the table with the decision label

    """
    decision_table = np.argmax((w0, w1), axis=0)
    return decision_table


def normalize_table(w0, w1):
    """ This is just a utility function to normalize the table

    Parameters
    ----------
    w0 : array-like of shape = [n, 1]
        Label 0 frequency table.

    w1 : array-like of shape = [n, 1]
        Label 1 frequency table.

    Returns
    -------
    w0 : array-like of shape = [n, 1]
        Label 0 frequency table. Normalized though.

    w1 : array-like of shape = [n, 1]
        Label 1 frequency table. Normalized though.

    """
    total = float(np.sum([w0, w1]))
    return  w0/total, w1/total

def apply_feature_selection(data, subset, w0, w1):
    """This function performs one iteration of the learning algorithm, given 
    as input the indices of the features selected for this iteration.
    The decision table for the classifier and the classification error for 
    this iteration are returned.
    As a byproduct, it also updates the weight vectors w0 and w1,
    associated with (input pattern, desired label) combinations.
    """
    subset_data = data[:, subset]
    hdata = _product_hash(subset_data)
    cls_error, updated_decision = _apply_transform(hdata, w0, w1)
    beta = cls_error/(1 - cls_error)
    w0, w1 = update_weights(w0, w1, beta, updated_decision)
    return w0, w1, updated_decision, cls_error
    

def update_weights(w0, w1, beta_t, dec_table): #beta
    """ This is a utility function to update the 
    weights associated with pairs (training pattern, desired label),
    given as inputs the beta_t value and the updated decision table.
    
    Note that the weights are updated in place, so no other function should
    access the weight arrays while this function is running.

    Parameters
    ----------
    w0 : array-like of shape = [n, 1]
        Normalized frequency of label 0.        

    w1 : array-like of shape = [n, 1]
        Normalized frequency of label 1.

    beta_t: double value.
        Multiplicative factor for updating the frequency of input patterns
        that are correctly classified according to the input decision table.

    dec_table : array-like of shape = [n,1].
        The input decision table.

    Returns
    -------
    w0 : array-like of shape = [n, 1]
        Updated and normalized frequency of label 0.        

    w1 : array-like of shape = [n, 1]
        Updated and normalized frequency of label 1.        

    """
    update_w0 = (dec_table == 0)
    w0[update_w0] *= beta_t
    w1[~update_w0] *= beta_t
    Z = w0.sum() + w1.sum()
    w0 /= Z
    w1 /= Z
    return w0, w1

def _apply_transform(hashed_table, w0, w1):
    """
    
    Parameters:
    ---------------
    hashed_table: array-like of shape = [r, 1].
        Each row a hashed version of an input pattern, so we can apply 'unique'
        directly to this array.
    w0: array-like of shape = [r, 1].
    w1: array-like of shape = [r, 1].
    
    Returns:
    ---------------
    cls_error: classification error for current iteration.
    updated_decision: decision table for current classifier.
    
    """
    # TODO: think of a better name for this function
    unique_groups, inverse_index = np.unique(hashed_table, 
                                             return_inverse=True)
    sum0 = []                                         
    sum1 = []
    for g in unique_groups:
        sum0.append(w0[hashed_table == g].sum())
        sum1.append(w1[hashed_table == g].sum())
    cls_error = error(sum0, sum1)
    hashed_decision = make_decision(sum0, sum1)
    updated_decision = hashed_decision[inverse_index]
    return cls_error, updated_decision

def _product_hash(table):
    """
    This function hashes each row of the input table by means of a dot
    product with a vector of coefficients. Potentially very slow.
    
    
    Parameters:
    -------------
    table: array-like of shape = [r, s].
        Each row of the table is a binary pattern (e.g. [0, 0, 1, 0, 1]).
        
    
    Returns:
    -------------
    hashed_table: array-like of shape = [r, 1].    
       
    """
    _, ncols = table.shape
    coeff = 2**np.arange(ncols)
    hashed_table = np.dot(table, coeff)
    return hashed_table
    