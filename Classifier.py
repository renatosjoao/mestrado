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


XPL_file_path = '/home/rsjoao/Dropbox/projetoMestrado/codigo/DRIVE/training set/drive7'
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
        The error value for the current iteraction.

    """
    epsilon_t = 0.0
    for a,b in zip(w0,w1):
        if a < b:
            epsilon_t = epsilon_t + a
        elif b < a:
            epsilon_t = epsilon_t + b
        else:
            epsilon_t = epsilon_t + a
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
    w0 = freq0/freq_sum(freq0,freq1)
    w1 = freq1/freq_sum(freq0,freq1)
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
    Taux = np.array(())
    for a,b in zip(w0,w1):
        if a > b:
            Taux =  np.append(Taux,[0])
            #Taux[i] = np.hstack((row,0))
        if a < b:
            Taux =  np.append(Taux,[1])
           #Taux[i] = np.hstack((row,1))
        if a == b:
            Taux =  np.append(Taux,[0])
           #Taux[i] = np.hstack((row,0))
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
    return  w0/np.sum([w0,w1]), w1/np.sum([w0,w1])

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
    of weights given the alpha_t value

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
            w1[t] = w1[t] * 1
        else:
            w1[t] = j * betha_t
            w0[t] = w0[t] * 1
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
    uniq = np.unique(Table.view(Table.dtype.descr * Table.shape[1]))
    unique = uniq.view(Table.dtype).reshape(-1, Table.shape[1])
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

if __name__ == "__main__":
    #main()
    Matriz =  read_from_XPL(XPL_file_path)
    freq0 = Matriz.freq0.astype('double')
    freq1 = Matriz.freq1.astype('double')
    [w0,w1] = create_freq_Table(freq0,freq1)

    #########sel_car#######

    #for i in range(20):
    #    decTable = make_decision(w0,w1)
    #    epsilon_t = error(w0,w1)
    #    betha_t = betha_factor(epsilon_t)
    #    [w0,w1] = update_Table(w0, w1, betha_t, decTable)
    #    [w0,w1] = normalize_Table(w0, w1)

Table = np.array([[0, 0, 0, 0], [1, 1, 1, 0], [1, 0, 0, 1],[0, 0, 0, 0], [0, 0, 0, 1],
                  [1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 1, 1],[0, 1, 0, 0], [0, 1, 0, 0],
                  [1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1],[0, 1, 0, 0], [0, 1, 1, 0],
                  [1, 1, 0, 0], [1, 1, 1, 1], [0, 0, 1, 1],[0, 1, 0, 1], [0, 1, 0, 1] ])

a = np.array([[1, 1],[0, 0],[0, 0],[0, 0],[0, 0],[0, 1],[0, 1],[0, 1],[1, 0],[0, 1],
              [0, 1],[0, 0],[0, 1],[0, 1],[0, 1],[0, 1],[1, 0],[0, 0],[0, 0],[1, 0],
              [1, 0],[0, 0],[0, 0],[0, 1],[0, 1],[1, 0],[0, 0],[0, 0],[0, 1],[1, 0],
              [0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 0],[0, 1],[0, 1],[1, 0],
              [0, 1],[0, 0],[0, 1],[0, 1],[0, 1],[1, 0],[1, 0],[0, 0],[0, 0],[1, 0]])
#print unique_rows(Table)
#print np.array([np.array(x) for x in set(tuple(x) for x in a)]) # or "list(x) for x in set[...]"
#b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
#_, idx = np.unique(b, return_index=True)
#unique_a = a[idx]
#print idx





#print aux
#print meHash
#for linha in uniq:
#    for row in a:
#        if tuple(row) == tuple(linha):
#            aux = myhash.get(tuple(row))
#            aux.append(index(linha))
#dict = {'Name': 'Zara', 'Age': 7, 'Class': 'First'};


#arrayList.append(99)
#dict['Age'] = arrayList
#arrayList.append(98)
#arrayList.append(97)
#dict['Age'] = arrayList
#arrayList = arrayList.append(2)
#dict['Age'] = arrayList
#dict['School'] = "DPS School"; # Add new entry



#print "dict['Age']: ", dict['Age'];

#subset = np.array([0,2,4,6,8,10,6,11,12,6,13,6,14,15,16,17,18,19,20])
#itemindex = np.where(a == tuple([1,1]) )
#print np.nonzero(a == tuple([1,1]))
#print itemindex

#x = np.array([0,0,0,8,5,0,1,0,1,0,1,3,4,5,3,4,5,3,4,5,6,7,8,7,6,8])
#print np.bincount(x)
#xx =  np.unique(x,True)
#print xx
#print subset
#print

#w0 = np.array([0.2, 0.1, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0.1])
#print w0
#groups = np.array([0,0,0,0,1,1,1,0,2])
#print groups
#unique_groups = np.unique(groups,True)
#print unique_groups

#sums = np.histogram(groups,bins=np.arange(groups.min(), groups.max()+2),weights=w0)[0]
#print sums
#freqTable = create_freq_Table(Matriz_t1)
#print freqTable
#error_t = error(freqTable,0)
#print alpha_factor(error_t)
#betha = betha_factor(error_t)
#print betha_factor(error_t)



    #for t in range(10):
    #    print ""
#print Table[:,:-2]
#print np.min((row[-2],row[-1]))
#print np.append(Table[0],[4,5,6])

#TODO:
#def create_empy_hash_table():
#    """ This is a very simple function to create an empty hash table """
#    my_hash = {}
#    return my_hash
#
#TODO:
#def read_from_csv(CSV_file_path):
#    """ Utility function to read w-pattern from csv file and return a numpy array"""
#    for row in CSV_file_path:
#        print row
#    return 0
#
#TODO:
#def dec_from_matrixRow(matrixRow):
#    """ Utility function to concatenate the values from a matrix row and convert it to int
#
#    :Input:
#     `matrixRow` : [0,0,1,1]
#
#    :Return: 3
#    """
#    string = ''
#    for i in matrixRow:
#        string +=`i`
#    dec = int(string,2)
#    return dec
#
#TODO:
#def build_dict(matrix_with_freq):
#    """Utility function to create a hash table from matrix previously appended with freq0 and freq1
#
#    :Input:
#     `matrix_with_freq` : Matrix with freq0 and freq1 columns appended [0, 1, 1, 2, 1]
#
#    :Return: dictionary {3: array([0, 1, 1, 2, 1], dtype=int32)}
#    """
#    dict = create_empy_hash_table()
#    for row  in matrix_with_freq:
#        dict[dec_from_matrixRow(row[:-2])] = row
#    return dict


####You use the built-in int() function, and pass it the base of the input number, i.e. 2 for a binary number.
#print Matriz_t1[0]
#MM = Matriz_t1.astype('double')
#Matriz_t1[:,-1]/freq_sum(Matriz_t1)[0]
#MM[:,-1] = Matriz_t1[:,-1]/freq_sum(Matriz_t1)[1]
#MM[:,-2] = Matriz_t1[:,-2]/freq_sum(Matriz_t1)[0]
#Matriz_t1[:,-1] = Matriz_t1[:,-1]/freq_sum(Matriz_t1)[0]
#Matriz_t1 = Matriz_t1.astype('double')
#print Matriz_t1[:,-2]
#print MM[8]

#for testing purposes
#Table = np.array(([0,0,0.1,0.2],
#               [0,1,0.075,0.225],
#               [1,0,0.1,0.075],
#               [1,1, 0.125, 0.1]))
#
#print alpha_factor(error(Table,0))
#for row in Matriz_t1:
#    print row
#print Matriz_t1
#print Matriz_t1[:,-1]/w1Sum
#print Matriz_t1[:,-2]/w0Sum
#[a1,a2] = freq_sum(Matriz_t1)
#print a1,a2
#print Matriz_t1
#print dec_from_matrixRow(Matriz.data[2])
#build_dict()

#def unique_rows(data):
#    unique = dict()
#    for row in data:
#        row = tuple(row)
#
#        if row in unique:
#            unique[row] += 1
#        else:
#            unique[row] = 1
#    return unique

#@deprecated
#def append_to_matrix(matrix1,matrix1_ncolumn, freq0, freq1):
#    """ This function is meant to gather the window patterns and labels (0,1)
#    frequencies into a single table.  Each row from freq0 matrix is appended to
#    each respective row from matrix1 and each row from freq1 is appended to
#    each respective row from matrix1.
#
#   Parameters
#   ----------
#   matrix1 : It is supposed to be a matrix read from xpl, i.e. matrix1 = matrix_read.data
#   matrix1_ncolumn : It represents the number of columns from matrix1, i.e. matrix_read.data.shape[1]
#   freq0 : freq0 is a matrix of labels 0 frequencies, i.e.  matrix_read.freq0
#   freq1 : freq1 is a matrix of labels 1 frequencies, i.e.  matrix_read.freq1
#
#   Returns
#   -------
#   : numpy matrix with appended columns of freq0 and freq1
#
#    >>> Wpattern, freq0, freq1
#    >>> '0,0,0,0, 200, 50'
#
#   """
#    ###NOTE: data read from xplutil is uint8. Must convert to int64 or append wont work
#    matrix1 = matrix1.astype('int64')
#    freq0 = freq0.astype('int64')
#    freq1 = freq1.astype('int64')
#    aux_matrix = np.insert(matrix1,matrix1_ncolumn,values=freq0,axis=1)
#    aux_matrix = np.insert(aux_matrix,matrix1_ncolumn+1,values=freq1, axis=1)
#    return aux_matrix

#TODO:
#def resample(Table, Subset):
#    """Predict classes for X.
#
#    The predicted class of an input sample is computed
#    as the weighted mean prediction of the classifiers in the ensemble.
#
#    Parameters
#    ----------
#    X : array-like of shape = [n_samples, n_features]
#        blabalbal.
#
#    Returns
#    -------
#    y : array of shape = [n_samples]
#
#    """
#    return 0

#def train_classifier():
#def test_classifier():
#def apply_classifier():    
#def fit(self, X, y):
#def predict(self,X):
#M = np.array([[0,0,0,3,5],[0,0,1,2,5],[0,1,0,3,2],[0,1,1,2,4],[1,0,0,3,1],[1,1,0,5,1],[1,1,1,1,0]])
#
#M
#
#array([[0, 0, 0, 3, 5],
#       [0, 0, 1, 2, 5],
#       [0, 1, 0, 3, 2],
#       [0, 1, 1, 2, 4],
#       [1, 0, 0, 3, 1],
#       [1, 1, 0, 5, 1],
#       [1, 1, 1, 1, 0]])
#M[:,(0,2)]