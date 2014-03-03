# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 15:37:13 2014

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
import imageset

def main(trainset, window, nfeatures, niterations, image, savetodir):
    output = savetodir+"temp.xpl"
    #building a XPL to start processing
    ensemble.build_xpl(trainset,window,output)
    XPL_filepath = output
    #now let's read the XPL
    XPL_data = xplutil.read_xpl(XPL_filepath)
    #writing the minimal empirical error to file
    ensemble.write_min_empirical_error(savetodir+"min_emp_err.txt",ensemble.min_empirical_error(XPL_data))
    #training the ensemble
    ensemble.train(XPL_data, nfeatures, niterations, savetodir)
    #building first level operators
    ensemble.build_operators(savetodir , niterations)
    #combining all the operators
    ensemble.build_operator_combination(trainset, np.array(range(niterations)), savetodir, savetodir+"twoLevel")

    #applying the first level operators on the given image
    for i in range(niterations):
        ensemble.apply_operator(savetodir+"mtm"+str(i)+"-op", image, savetodir+"mtm"+str(i)+"-op-files/image_processed")

    #applying the second level operator on the given image
    ensemble.apply_operator(savetodir+"twoLevel",image, savetodir+"twoLevel-files/level1/operator0/image_processed")

    #Writing MAE from training set  to file
    imgset = imageset.Imageset()
    set = imgset.read(trainset)
    mae_t = ensemble.generic_mae(set)
    file = open(savetodir+"MAE_TrainingSet.txt", "w")
    file.write(str(mae_t))
    file.close()

    #***      testing the Ensemble !!!
    #set = imageset.Imageset()
    #conjunto = set.read(testset)
    #mae_test = ensemble.generic_mae(conjunto)
    #file = open(dir_path+"MAE_TestSet.txt", "w")
    #file.write(str(mae_test))
    #file.close()


if __name__ == "__main__":
     parser = argparse.ArgumentParser(description="Performs a single iteration of the ensemble and apply the operator on the given image.")
     parser.add_argument("-t", "--trainset", help="Training set file path.")
     parser.add_argument("-w", "--window", help="Window file path.")
     parser.add_argument("-n", "--nfeatures", type=int, help="The number of features to be selected.")
     parser.add_argument("-i", "--niterations", type=int, help="The number of iterations to run the ensemble training.")
     parser.add_argument("-img", "--image", help="Image file to apply the operator." )
     parser.add_argument("-s", "--savetodir", help="Directory to save files.")

     args = parser.parse_args()

     if not args.trainset:
         print "Provide the training set. -t --trainset"
     if not args.window:
         print "Provide the window file.  -w --window"
     if not args.nfeatures:
         print "Provide the number of features to be selected.  -n --nfeatures"
     if not args.niterations:
         print "Provide the number of iterations to run the ensemble algorithm.  -i --niterations"
     if not args.image:
         print "Provide image file to apply the operator.   -img --image"
     if not args.savetodir:
         print "Provide directory to save files.  -s --savetodir"
     else:
        main(args.trainset, args.window, args.nfeatures, args.niterations, args.image, args.savetodir)