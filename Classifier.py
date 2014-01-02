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
import sys
import time


XPL_file_path = ''
CSV_file_path= ''


def read_from_XPL(XPL_file_path):
    """xplutil.py already reads from XPL file.

    Parameters
    ----------
    XPL_file_path : '/home/jdoe/path_to_xpl'
                    The path to the XPL file.

    Returns
    -------
    result : ExampleData(data, freq0, freq1, winshape, windata, filename)
            Same as xplutil returns.

    """
    result = xplutil.read_xpl(XPL_file_path)
    return result

def freq_sum(w0, w1):
    """ This function is meant to sum all the elements from w0 and  w1.

    Parameters
    ----------
    w0 : array-like of shape = [n, 1]
        Label 0 frequency table.

    w1 : array-like of shape = [n, 1]
        Label 1 frequency table.

    Returns
    -------
    np.sum([w0,w1]) : double value
        Returns the sum for all w0,w1 values.

    """
    return np.sum([w0,w1])

def error(w0, w1):
    """ This is a function to calculate the error for the current interaction.
    Error is calculated on grouped weight.
    Note the error is always calculated after feature selection.

    Parameters
    ----------
    w0 : array-like of shape = [n, 1]
        Label 0 frequency table.

    w1 : array-like of shape = [n, 1]
        Label 1 frequency table.

    Returns
    -------
    epsilon_t : double
        The error value for the current iteraction.

    """
    epsilon_t = 0.0
    for a,b in zip(w0,w1):
        if a <= b:
            epsilon_t = epsilon_t + a
        else:
            epsilon_t = epsilon_t + b
    return epsilon_t

def betha_factor(epsilon_t):
    """ This is a function to calculate the betha_t factor

    Parameters
    ----------
    epsilon_t : double
        Error for the current iteraction.

    Returns
    -------
     betha_t : double
         Betha value for the current iteraction

     """
    betha_t = epsilon_t / (1.0 - epsilon_t)
    return betha_t

def create_freq_Table(freq0, freq1):
    """ This is a function to create a frequency table.

    Parameters
    ----------
     freq0 : array-like of shape = [n, 1]
       It is the original matrix with freq0.
     freq1 : array-like of shape = [n, 1]
       It is the original matrix with freq1.

    Returns
    -------
    w0 : array-like of shape = [n, 1]
        Label 0 frequency table.

    w1 : array-like of shape = [n, 1]
        Label 1 frequency table.

    """
    fsum = freq_sum(freq0,freq1)
    w0 = freq0/fsum
    w1 = freq1/fsum
    return w0, w1

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
    Taux :  array-like of shape = [n, 1]
        This is the table with the decision label

    """
    Taux = np.argmax((w0,w1),axis=0)
    return Taux


def normalize_Table(w0,w1):
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

    #total = np.sum(Table[:,[-2,-1]])
    #for row in Table:
    #    row[-1] = row[-1]/total
    #    row[-2] = row[-2]/total
    total = np.sum([w0,w1])
    return  w0/total, w1/total

#TODO:
def sel_car(Table, w0, w1):
    """ A function to feature selection procedure
    As it is not implemented yet it is returning a pre-set numpy array of indexes.
    subset = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

    Parameters
    ----------

    Returns
    -------
    """
    subset = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    return subset

def update_Table(w0,w1,betha_t,decTable):#betha
    """ This is just a utility function to update the table
    of weights given the betha_t value

    Parameters
    ----------
    w0 : array-like of shape = [n, 1]
        Label 0 frequency table.

    w1 : array-like of shape = [n, 1]
        Label 1 frequency table.

    betha_t: double value.
        That value will be used to update the weights in the
        frequency table.

    decTable : array-like of shape = [n,1].
        The input decision table.

    Returns
    -------
    w0 : array-like of shape = [n, 1]
        Label 0 frequency table updated.

    w1 : array-like of shape = [n, 1]
        Label 1 frequency table updated.

    """
    t = 0
    for i,j,k in zip(w0,w1,decTable):
        if k == 0:
            w0[t] = i * betha_t
        else:
            w1[t] = j * betha_t
        t = t +1
    return w0, w1

def unique_rows(Table):
    """ This is a utility function to return unique rows in a given table.

    Parameters
    ----------
    Table : array-like of shape = [n,m].
        The input table with repeated rows.

    Returns
    -------
    unique : array-like of shape = [n,m].
        The output table with unique rows.

    """
    b = Table.ravel().view(np.dtype((np.void, Table.dtype.itemsize*Table.shape[1])))
    _, unique_idx = np.unique(b, return_index=True)
    unique = Table[np.sort(unique_idx)]
    return unique

def create_dictionary(Table):
    """ This function searches for repeated rows in a table,
    remove repeated rows and then creates a dictionary with the rows and indexes.

    Parameters
    ----------
    Table : array-like of shape = [n,m].
        The output table with unique rows.

    Returns
    -------
    dic : dictionary type.
        It  returns a dictionary type with keys being the patterns and values being an array with the indexes
        for all of the pattern occurrences.
        For example : {(1, 0): (array([ 8, 16]),), (0, 0): (array([ 47, 48]),), (1, 1): (array([0]),)}

    """
    uniq =  unique_rows(Table)
    dic = {}
    for row in uniq:
        dic[tuple(row)] = np.where(np.all(row==Table,axis=1))
###########   index = np.where(np.all(row == a,axis=1))
    return dic

def create_projected_tab(Table, subset_index_Array):
    """ This function is meant to create a projected table given the subset index array.

    Parameters
    ----------
    Table : array-like of shape = [n,m].
        The input table with patterns rows.

    subset_index_Array : array-like of shape = [1,m].
        The indexes selected from the feature selection process.

    Returns
    -------
    newTable : array-like of shape = [n,m].
        The output projected table with only the selected columns.

    """
    newTable = Table[:,subset_index_Array]
    return newTable

def group_weights(dic,uniq, w0, w1):
    """ This function is meant to group weights from w0 and w1 based on the
    indexes passed by the dictionary.

    Parameters
    ---------
    dic : dictionary type.
        A dictionary type with keys being the patterns and values being an array with the indexes
        for all of the pattern occurrences.
        For example : {(1, 0): (array([ 8, 16]),), (0, 0): (array([ 47, 48]),), (1, 1): (array([0]),)}

    uniq : array-like of shape = [n,m].
        The table with unique patterns.

    w0 : array-like of shape = [n, 1]
        Label 0 frequency table.

    w1 : array-like of shape = [n, 1]
        Label 1 frequency table.

    Returns
    -------
    w0 : array-like of shape = [n, 1]
        Label 0 frequency table.

    w1 : array-like of shape = [n, 1]
        Label 1 frequency table. Normalized though.

    """
    t=0
    waux0 = np.zeros((uniq.shape[0]))
    waux1 = np.zeros((uniq.shape[0]))
    for row in uniq:
        arr = dic.get(tuple(row.reshape(1,-1)[0]))
        indexes =  tuple(arr[0].reshape(1,-1)[0])
        waux0[t] = np.sum(w0[[[indexes]]])
        waux1[t] = np.sum(w1[[[indexes]]])
        t+=1
    w0 = waux0
    w1 = waux1
    return w0,w1

def unproject(dictionary,originalTable, uniqueRows, decTable):
    """  This function is meant to assign predictions from decTable to the table
    before resampling

    Parameters
    ---------
    dictionary : dictionary type.
        A dictionary type with keys being the patterns and values being an array with the indexes
        for all of the pattern occurrences.
        For example : {(1, 0): (array([ 8, 16]),), (0, 0): (array([ 47, 48]),), (1, 1): (array([0]),)}

    uniqueRows : array-like of shape = [n,m].
        The table with unique patterns.

    Returns
    -------
    decisionTable : array-like of shape = [n, 1]
        This  is supposed to be the decision table for the initial table.
    """
    decisionTable  = np.zeros((originalTable.shape[0],1))
    print decisionTable.shape
    i= 0
    for row in uniqueRows:
        arr = dictionary.get(tuple(row.reshape(1,-1)[0]))
        indexes =  tuple(arr[0].reshape(1,-1)[0])
        decisionTable[[indexes]] = decTable[i]
        i+=1
    return decisionTable

if __name__ == "__main__":
    #main()
    t0 = time.clock()
    param = sys.argv[1]
    #param = "/home/rsjoao/Dropbox/projetoMestrado/codigo/DRIVE/training set/drive5x5.xpl"
    print param
#************* PASSOS de EXECUCAO do ALGORITMO ************
    Matriz =  read_from_XPL(param)
    freq0 = Matriz.freq0.astype('double')
    freq1 = Matriz.freq1.astype('double')
    [w0,w1] = create_freq_Table(freq0,freq1)
    #########sel_car() #######
    Table = Matriz.data
    indexes = np.array([0,2,4,5,6,7,8,9,10,11,14,15,16,17,19,20,21,24])

    print time.clock() - t0, "seconds process time"

    for i in range(20):

        t0 = time.clock()
        projTable = create_projected_tab(Table, indexes)
        dict = create_dictionary(projTable)
        table_unique_rows = unique_rows(projTable)

        [w0_grp,w1_grp] = group_weights(dict, table_unique_rows,w0,w1)
        decTable = make_decision(w0_grp,w1_grp)
        epsilon_t = error(w0_grp,w1_grp)
        betha_t = betha_factor(epsilon_t)
        psiTable = unproject(dict, Table, table_unique_rows,decTable)
        [w0,w1] = update_Table(w0, w1, betha_t, psiTable)
        [w0,w1] = normalize_Table(w0, w1)
        print time.clock() - t0, "seconds process time"
