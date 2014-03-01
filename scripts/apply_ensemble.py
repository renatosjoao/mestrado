# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 21:18:09 2014

@author: rsjoao
"""
import os, sys
import argparse
thisdir = os.path.dirname(__file__)
libdir = os.path.join(thisdir, '../')

if libdir not in sys.path:
    sys.path.insert(0, libdir)

import classifier as cl
import ensemble
import xplutil
import feature as ft
import trioswindow as tw
import numpy as np

#This script runs 1 iteration from the original algorithm
#It saves a XPL file, a MTM file, an operator and applies the operator
#on the image passed by argument.

def main(trainset, window, nfeatures, image, savetodir):
    output = savetodir+"temp.xpl"
    #building the xpl file
    ensemble.build_xpl(trainset,window,output)
    #reading the "just" created xpl file
    result = xplutil.read_xpl(output)
    XPL_data = result.data
    w0 = result.freq0.copy()
    w1 = result.freq1.copy()
    #normalizing the table frequency values
    w0,w1 = cl.normalize_table(w0, w1)
    #calculating features and indexes
    indices, feature_list, _ = ft.cmim(XPL_data, w0, w1, nfeatures)
    #saving window file
    tw.to_window_file(indices, result.winshape, savetodir+"window.win")
    #applying the feature selection algorithm
    w0, w1, updated_decision, cls_error =  cl.apply_feature_selection(XPL_data, indices, w0, w1)
    #calculating the unique array and their indexes
    indices = np.sort(indices)
    unique_array, unique_index = cl._apply_projection(XPL_data, indices)
    #writing a mimterm file to disk
    xplutil.write_minterm_file(savetodir+"mintermfile.mtm",indices, result.winshape, unique_array,updated_decision[unique_index])
    #building the operator based on the mintermfile
    ensemble.build_operator(savetodir+"window.win", savetodir+"mintermfile.mtm", savetodir+"operator")
    #building a new XPL according to the learned window
    output = savetodir+"Learned.xpl"
    ensemble.build_xpl(trainset,savetodir+"window.win",output)
    result_img = savetodir+"Image_applied"
    #applying the operator on the image
    ensemble.apply_operator(savetodir+"operator", image, result_img)


if __name__ == "__main__":
     parser = argparse.ArgumentParser(description="Performs a single iteration of the ensemble and apply the operator on the given image.")
     parser.add_argument("-t", "--trainset", help="Training set file path.")
     parser.add_argument("-w", "--window", help="Window file path.")
     parser.add_argument("-n", "--nfeatures", type=int, help="The number of features to be selected.")
     parser.add_argument("-i", "--image", help="Image file to apply the operator." )
     parser.add_argument("-s", "--savetodir", help="Directory to save files.")

     args = parser.parse_args()

     if not args.trainset:
         print "Provide the training set. -t --trainset"
     if not args.window:
         print "Provide the window file.  -w --window"
     if not args.nfeatures:
         print "Provide the number of features to be selected.  -n --nfeatures"
     if not args.image:
         print "Provide image file to apply the operator.   -i --image"
     if not args.savetodir:
         print "Provide directory to save files.  -s --savetodir"
     else:
        main(args.trainset, args.window, args.nfeatures, args.image, args.savetodir)
