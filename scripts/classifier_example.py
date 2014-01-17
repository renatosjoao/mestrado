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

def main(xpl_data, num_features=None, num_iterations=None):
    height, width = xpl_data.winshape    
    data = xpl_data.data 
    w0 = xpl_data.freq0
    w1 = xpl_data.freq1
    error_list = []
    DEC = np.zeros(w0.shape)
    GVector = []
    i=0
    winfile = "window_"
    win = ".win"
    png = ".png"

    for i in range(num_iterations):

        indices, feature_list, _ = ft.cmim(data, w0, w1, num_features)

        tw.to_window_file(indices, xpl_data.winshape, winfile+str(i)+win)

        tw.to_image_file(indices,xpl_data.winshape, winfile+str(i)+png, scale=8)

        w0, w1 = cl.normalize_table(w0, w1)

        w0, w1, updated_decision, cls_error =  cl.apply_feature_selection(data, indices, w0, w1)

        error_list.append(cls_error)

        bt = cl.beta_factor(cls_error)

        gam = np.log(1/bt)
        GVector = np.append(GVector,gam)

        #DEC represents the Decision Table. Each column represents the decision
        #for an iteration
        DEC = np.column_stack((DEC,updated_decision))

    #Must delete the first column because it contains only Zeros as it was initialized with np.zeros()
    DEC = np.delete(DEC,0, axis=1)

    Hip = cl._final_Hip(DEC, GVector)

        
if __name__ == "__main__":
     parser = argparse.ArgumentParser(description="Perform t iterations of the current ensemble algorithm on a given XPL file.")

     #parser.add_argument("-w", "--window", help="Trios window file name where to save the obtained window")

     parser.add_argument("-n", "--numfeatures", type=int, help="Maximum number of features")

     parser.add_argument("-t", "--numiterations", type=int, help="Number of iterations to run the algorithm.")

     parser.add_argument("filename", help="XPL filename")
     
     args = parser.parse_args()
     
     if args.filename:
         xpl_data = xplutil.read_xpl(args.filename)    
         
         main(xpl_data, args.numfeatures, args.numiterations)
              
     else:
         print "Must provide input file name."
        