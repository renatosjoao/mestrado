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

def main(trainset, testset, window, nfeatures, niterations, savetodir):
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
    print
    print "... Building first level operators ...\n"
    ensemble.build_operators(savetodir , niterations)

    #creating the operators combinations
    for i in range(1,niterations):
        print
        print "...Building operators combination : 0 to %s ... \n" %str(i)
        ensemble.build_operator_combination(trainset, np.array(range(i+1)), savetodir, savetodir+"twoLevel_0_to_"+str(i))

    #combining all the operators
    #ensemble.build_operator_combination(trainset, np.array(range(niterations)), savetodir, savetodir+"twoLevel")

    #Writing MAE from training set  to file.
    set = imageset.Imageset()
    trainimgset = set.read(trainset)
    mae_t = ensemble.mae(trainimgset)
    file = open(savetodir+"MAE_TrainingSet.txt", "w")
    file.write(str(mae_t))
    file.close()

    #Writing MAE  from test set to file.
    set = imageset.Imageset()
    testimgset = set.read(testset)
    mae_test = ensemble.mae(testimgset)
    file = open(savetodir+"MAE_TestSet.txt", "w")
    file.write(str(mae_test))
    file.close()

    img_list = _get_imgList(testimgset)

    #applying the first level operators on the test set images
    for i in range(niterations):
        for j in img_list:
            ensemble.apply_operator(savetodir+"mtm"+str(i)+"-op", j, savetodir+"mtm"+str(i)+"-op-files/"+j.split("/")[-1][:-3]+"proc.pnm")
        ensemble.trios_test(savetodir+"mtm"+str(i)+"-op", testset, savetodir+"mtm"+str(i)+"-op-files/MAE.txt")

    #applying the second level operators combinations on the test set images
    for i in range(1,niterations):
        for j in img_list:
            ensemble.apply_operator(savetodir+"twoLevel_0_to_"+str(i),j, savetodir+"twoLevel_0_to_"+str(i)+"-files/level1/operator0/"+j.split("/")[-1][:-3]+"proc.pnm")
        ensemble.trios_test(savetodir+"twoLevel_0_to_"+str(i), testset, savetodir+"twoLevel_0_to_"+str(i)+"-files/level1/operator0/MAE.txt")

    #applying the second level operator on the given image
    #ensemble.apply_operator(savetodir+"twoLevel",image, savetodir+"twoLevel-files/level1/operator0/image_processed")

    # MAE_TrainingSet and MAE_TestSet are generic values. Consider as if the identity operator
    # had been applied to the images. It is only a reference value for improvement.


def _get_imgList(imageset):
    """The ensemble needs a list of images to apply the operator.
    The list is read from the test set file and the ideal images are discarded here.
    It is only meant to return a list of input images to be processed by the operators.
    """
    img_list = []
    num = len(imageset)
    for i in range(num):
        img_list.append(imageset[i][0])
    return img_list

if __name__ == "__main__":
     parser = argparse.ArgumentParser(description="Performs n iterations of the ensemble and apply the operator on the given image.")
     parser.add_argument("-tr", "--trainset", help="Training set file path.")
     parser.add_argument("-te", "--testset", help="Test set file path.")
     parser.add_argument("-w", "--window", help="Window file path.")
     parser.add_argument("-n", "--nfeatures", type=int, help="The number of features to be selected.")
     parser.add_argument("-i", "--niterations", type=int, help="The number of iterations to run the ensemble training.")
     parser.add_argument("-s", "--savetodir", help="Directory to save files.")

     args = parser.parse_args()

     if not args.trainset:
         print "Provide the training set. -tr --trainset"
         raise Exception('Missing argument')
     if not args.testset:
         print "Provide the test set. -te --testset"
         raise Exception('Missing argument')
     if not args.window:
         print "Provide the window file.  -w --window"
         raise Exception('Missing argument')
     if not args.nfeatures:
         print "Provide the number of features to be selected.  -n --nfeatures"
         raise Exception('Missing argument')
     if not args.niterations:
         print "Provide the number of iterations to run the ensemble algorithm.  -i --niterations"
         raise Exception('Missing argument')
     if not args.savetodir:
         print "Provide directory to save files.  -s --savetodir"
         raise Exception('Missing argument')
     else:
        main(args.trainset, args.testset, args.window, args.nfeatures, args.niterations, args.savetodir)
