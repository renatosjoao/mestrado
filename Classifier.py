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
    

#TODO:
def read_from_csv(CSV_file_path):
    """ Utility function to read w-pattern from csv file and return a numpy array"""
    for row in CSV_file_path:
        print row
    return 0


def freq_sum(M):
    """Frequency sum """
    aux = np.sum(M,axis=0)
    return [aux[-1],aux[-2]]

     
    
def append_to_matrix(matrix1,matrix1_ncolumn, matrix2, matrix3):
    """ Each row of matrix2 is appended to each respective row of matrix1
        and each row of matrix3 is appended to each respective row of matrix1

   :Input:
    `matrix1`: supposed to be a matrix read from xpl.  >>> matrix1 = matrix_read.data
    `matrix1_ncolumn`: matrix1 number of columns.  >>> matrix_read.data.shape[1]
    `matrix2`: matrix2 is the matrix of frequencies of labels 0. >>> matrix2 = matrix_read.freq0
    `matrix3`: matrix3 is the matrix of frequencies of labels 1. >>> matrix3 = matrix_read.freq1

   :Return: numpy matrix with appended columns of freq0 and freq1

    >>> data read from xplutil is uint8. Must convert to int32 or append wont work
        
     """
    matrix1 = matrix1.astype('int32')
    matrix2 = matrix2.astype('int32')
    matrix3 = matrix3.astype('int32')
    aux_matrix = np.insert(matrix1,matrix1_ncolumn,values=matrix2,axis=1)
    aux_matrix = np.insert(aux_matrix,matrix1_ncolumn+1,values=matrix3, axis=1)
    return aux_matrix


def create_empy_hash_table():
    """ This is a very simple function to create an empty hash table """
    my_hash = {}
    return my_hash

def dec_from_matrixRow(matrixRow):
    """ Utility function to concatenate the values from a matrix row and convert it to int 
    
    :Input:
     `matrixRow` : [0,0,1,1]
     
    :Return: 3
    """
    string = ''
    for i in matrixRow:
        string +=`i`
    dec = int(string,2)
    return dec


def build_dict(matrix_with_freq):
    """Utility function to create a hash table from matrix previously appended with freq0 and freq1
    
    :Input:
     `matrix_with_freq` : Matrix with freq0 and freq1 columns appended [0, 1, 1, 2, 1]
    
    :Return: dictionary {3: array([0, 1, 1, 2, 1], dtype=int32)}
    """
    dict = create_empy_hash_table()
    for row  in matrix_with_freq:
        dict[dec_from_matrixRow(row[:-2])] = row
    return dict



####You use the built-in int() function, and pass it the base of the input number, i.e. 2 for a binary number.


Matriz =  read_from_XPL(XPL_file_path)
dic = build_dict(append_to_matrix(Matriz.data,Matriz.data.shape[1],Matriz.freq0,Matriz.freq1))
print dic
#print dec_from_matrixRow(Matriz.data[2])
#build_dict()




#TODO:
def error(t):
    """ This is a function to calculate the error for the t interaction """
    #epsilon_t = SUM D_t(x_i) I(y_i != h_t(x_i))
    #return epsilon_t

#TODO:    
def norm():
    """ This is a function to calculate the normalizing value """
    #norm =      
    #return norm
    #np.exp()

#TODO:    
def alpha_factor(epsilon_t):
    """ This is a function to calculate the alpha_t factor """
    alpha_t = 1/2*np.log((1-epsilon_t)/epsilon_t)
    return alpha_t

#TODO:
def resample():
    """ function to resample """
    return 0    

#def update_P():    

#def train_classifier():

#def test_classifier():

#def apply_classifier():    





def sel_car(Wpattern, w0, w1):
    """ A function to feature selection """
    
    return subset   


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
