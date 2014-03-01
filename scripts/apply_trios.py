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

import ensemble

#This script builds an image operator
#It saves a XPL file, a MTM file, an operator and applies the operator
#on the image passed by argument.

def main(trainset, window, image, savetodir):
    output = savetodir+"temp.xpl"
    fname = savetodir+"operator"
    #building the xpl file
    ensemble.build_xpl(trainset,window,output)
    #writing minterm file to disk
    ensemble.trios_build_mtm(window, trainset, savetodir+"mintermfile.mtm")
    #building an operator with trios_build tool
    ensemble.trios_build(window, trainset, fname)
    result_img = savetodir+"Image_applied"
    #applying the operator on the image
    ensemble.apply_operator(savetodir+"operator", image, result_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an Operator given a training set file and a window file and applies it on a given image.")
    parser.add_argument("-t", "--trainset", help="Training set file path.")
    parser.add_argument("-w", "--window", help="Window file path.")
    parser.add_argument("-i", "--image", help="Image file to apply the operator." )
    parser.add_argument("-s", "--savetodir", help="Directory to save files.")

    args = parser.parse_args()

    if not args.trainset:
        print "Provide the training set. -t --trainset"
    if not args.window:
        print "Provide the window file.  -w --window"
    if not args.image:
        print "Provide image file to apply the operator.   -i --image"
    if not args.savetodir:
        print "Provide directory to save files.  -s --savetodir"
    else:
       main(args.trainset, args.window, args.image, args.savetodir)