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
    """ xplutil.py already reads from XPL file.
    
    :Input:
        `file_path` : path to the xpl file
     
     Result: 
         `matrix_read` : numpy matrix read from XPL file
    """
    matrix_read = xplutil.read_xpl(XPL_file_path)    
    return matrix_read    
    
def freq_sum(Matrix):
    """ This function is meant to sum all the elements for the freq0 and freq1
    columns.

    :Input:
     `Matrix` : The input matrix must be in the following format
         [Wpattern, freq0, freq1]
         [0,0,0,0, 200, 50]
    :Output:
     `[freq0.sum(), freq1.sum()]` :

    """
    aux = np.sum(Matrix,axis=0)
    aux = aux.astype('double')
    return [aux[-2],aux[-1]]
    
def append_to_matrix(matrix1,matrix1_ncolumn, freq0, freq1):

    """ This function is meant to gather the window patterns and labels (0,1)
    frequencies into a single table.  Each row from freq0 matrix is appended to
    each respective row from matrix1 and each row from freq1 is appended to
    each respective row from matrix1.

   :Input:
    `matrix1`: It is supposed to be a matrix read from xpl, i.e. matrix1 = matrix_read.data
    `matrix1_ncolumn`: It represents the number of columns from matrix1, i.e. matrix_read.data.shape[1]
    `freq0`: freq0 is a matrix of labels 0 frequencies, i.e.  matrix_read.freq0
    `freq1`: freq1 is a matrix of labels 1 frequencies, i.e.  matrix_read.freq1

   :Return: numpy matrix with appended columns of freq0 and freq1


    >>> Wpattern, freq0, freq1
    >>> '0,0,0,0, 200, 50'

   """
    ###NOTE: data read from xplutil is uint8. Must convert to int64 or append wont work
    matrix1 = matrix1.astype('int64')
    freq0 = freq0.astype('int64')
    freq1 = freq1.astype('int64')
    aux_matrix = np.insert(matrix1,matrix1_ncolumn,values=freq0,axis=1)
    aux_matrix = np.insert(aux_matrix,matrix1_ncolumn+1,values=freq1, axis=1)
    return aux_matrix

def error(Table, t):
    """ This is a function to calculate the error for the t interaction

    :Input:
     `Table` : Matrix after feature selection with weights w0 and w1  i.e. [0, 1, 1, 0.075, 0.1]

    :Return:
     `epsilon_t`: This is the error value for the current iteraction
    """
    epsilon_t = 0.0
    for row in Table:
        #row[-2] is the column that represents w0
        #row[-1] is the column that represents w1
        epsilon_t = epsilon_t + np.min((row[-2],row[-1]))
        #epsilon_t = SUM D_t(x_i) I(y_i != h_t(x_i))
    return epsilon_t


def alpha_factor(epsilon_t):
    """ This is a function to calculate the alpha_t factor

    :Input:
     `epsilon_t : Error for the current iteraction

    :Return:
     `alphta_t` : It is the alpha factor for the current iteraction
    """
    alpha_t = 0.5*(np.log((1.0-epsilon_t)/epsilon_t))
    return alpha_t





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


Matriz =  read_from_XPL(XPL_file_path)
w0Sum =  Matriz.freq0.sum()
w0Sum = w0Sum.astype('double')
w1Sum =  Matriz.freq1.sum()
w1Sum = w1Sum.astype('double')
Matriz_t1 = append_to_matrix(Matriz.data,Matriz.data.shape[1],Matriz.freq0,Matriz.freq1)
Matriz_t1 = Matriz_t1.astype('double')




#for testing purposes
Table = np.array(([0,0,0.1,0.2],
                [0,1,0.075,0.225],
                [1,0,0.1,0.075],
                [1,1, 0.125, 0.1]))

print alpha_factor(error(Table,0))
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


#TODO:
#def norm():
#    """ This is a function to calculate the normalizing value """
    #norm =
    #return norm
    #np.exp()



#TODO:
def resample(Table, Subset):
    """ function to resample """
    return 0

#TODO:
#def update_Table():

#def train_classifier():

#def test_classifier():

#def apply_classifier():    



#def sel_car(Wpattern, w0, w1):
#    """ A function to feature selection """
#
#    return subset

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
