# -*- coding: utf-8 -*-
##########################################
#Renato Stoffalette Joao
#
##########################################

__author__ = "Renato Stoffalette Joao(renatosjoao@gmail.com)"
__version__ = "$Revision: 0.1 $"
__date__ = "$Date: 2014// $"
__copyright__ = "Copyright (c) 2013 Renato SJ"
__license__ = "Python"

import _mypath
import feature as ft
import trioswindow as tw
import xplutil
import classifier as cl
import argparse
import numpy as np
import pickle
import graphics as gra

def main(xpl_data, num_features=None, num_iterations=None, save_todir=None):
    data = xpl_data.data
    # copies frequency data as original frequencies are used towards the end to estimate training error
    w0 = xpl_data.freq0.copy()
    w1 = xpl_data.freq1.copy()
    error_list = []
    mae_list = []
    DEC = np.zeros(w0.shape)
    GVector = []
    i=0
    winfile = "window_"
    win = ".win"
    png = ".png"
    w0_train, w1_train = cl.normalize_table(xpl_data.freq0, xpl_data.freq1)
    file = open(save_todir+"Error.txt", "w")

    for i in range(num_iterations):
        indices, feature_list, _ = ft.cmim(data, w0, w1, num_features)
        tw.to_window_file(indices, xpl_data.winshape, save_todir+winfile+str(i)+win)
        tw.to_image_file(indices,xpl_data.winshape, save_todir+winfile+str(i)+png, scale=8)
        w0, w1 = cl.normalize_table(w0, w1)
        w0, w1, updated_decision, cls_error =  cl.apply_feature_selection(data, indices, w0, w1)
        unique_array, unique_index = cl._apply_projection(data, indices)
        xplutil.write_minterm_file(save_todir+"mtm_"+str(i),indices, xpl_data.winshape,unique_array,updated_decision[unique_index])
        str_to_file = "Classification error for iteration " + str(i) +" = "+ str(cls_error) +".\n"
        file.write(str_to_file)
        error_list.append(cls_error)
        bt = cl.beta_factor(cls_error)
        gam = np.log(1/bt)
        GVector = np.append(GVector,gam)
        #DEC represents the Decision Table. Each column represents the decision
        #for an iteration
        DEC = np.column_stack((DEC,updated_decision))
        aux_dec = DEC
        aux_dec = np.delete(aux_dec,0, axis=1)
        hypothesis = cl.adaboost_decision(aux_dec, GVector)
        MAE_t = cl.mae_from_distribution(hypothesis,w0_train, w1_train)
        mae_list = np.append(mae_list,MAE_t)
        str_to_file = "MAE for iteration " + str(i) +" = "+ str(MAE_t) +".\n\n"
        file.write(str_to_file)

    #Must delete the first column because it contains only Zeros as it was initialized with np.zeros()
    DEC = np.delete(DEC,0, axis=1)
    hypothesis = cl.adaboost_decision(DEC, GVector)
     
    MAE = cl.mae_from_distribution(hypothesis, w0_train, w1_train)
    str_to_file = "Final MAE = "+ str(MAE)
    file.write(str_to_file)
    #print MAE
    file.close()
    gra.plot_MAE_iter(np.array(range(num_iterations)), np.array(mae_list))

        
if __name__ == "__main__":
     parser = argparse.ArgumentParser(description="Perform t iterations of the ensemble algorithm on a given XPL file.")

     parser.add_argument("-n", "--numfeatures", type=int, help="Maximum number of features")

     parser.add_argument("-t", "--numiterations", type=int, help="Number of iterations to run the algorithm.")

     parser.add_argument("filename", help="XPL filename")
     
     parser.add_argument("-s", "--savetodir", help="Directory to save window files.")

     args = parser.parse_args()
     
     if args.filename:
         xpl_data = xplutil.read_xpl(args.filename)    
         
         main(xpl_data, args.numfeatures, args.numiterations, args.savetodir) 
              
     else:
         print "Must provide input file name."
        
def save_decision(decision_table, file_path):
    """ Meant to save decision table to file
    """
    np.savetxt(file_path, decision_table, fmt="%d")

def save_pattern(pattern_table, file_path):
    """ Meant to save pattern table to file
    """
    np.savetxt(file_path, pattern_table, fmt="%d")
