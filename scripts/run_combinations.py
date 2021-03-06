# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 11:54:17 2014

@author: rsjoao
"""

import os, sys
import argparse
thisdir = os.path.dirname(__file__)
libdir = os.path.join(thisdir, '../')

if libdir not in sys.path:
    sys.path.insert(0, libdir)

import ensemble
import trioswindow as tw
import numpy as np
import random
import subprocess
import shlex
import imageset

def _get_indexes(num_pixels, winshape):
    """ a function to return indexes randomly selected

    num_pixels is the number of pixels to be set to 1 on the window
    winshape is the window shape i.e.(11,11)
    """
    winh = winshape[0]
    winl = winshape[1]
    winsize = winh * winl
    numbers = range(0,winsize/2) + range((winsize/2)+1,winsize)
    r = random.sample(numbers, num_pixels -1)
    r.append(winsize/2)
    return np.sort(r)

def make_windows(savetodir, winshape, m, n_windows):
    """
    a function to make the windows randomly and save to dir
    savetodir is the directory to save the windows
    winshape is the window shape
    m is the number of pixels to be selected
    n_windows is the number of windows to save
    """
    for i in range(int(n_windows)):
        pixels = _get_indexes(m,winshape)
        tw.to_window_file(pixels, winshape, savetodir+"window_"+str(i)+".win")
        tw.to_image_file(pixels,winshape, savetodir+"window_"+str(i)+".png", scale=8)
    return 0

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

def build_operator(win, imgset,fname):
    """ This function is meant to build first level operators
    win - window file path
    imgset - image set file path
    fname - output operator file
    """
    ensemble.trios_build(win, imgset, fname)
    return 1

def build_operators_combination(imgset, op_to_combine, op_dir, fname):
    #op_to_combine i.e 1,2,3 or 2,3,4 or 3,4,5
    operators_list = []
    for i in op_to_combine:
        operators_list.append("window_"+str(i)+"_op")
    ops = " ".join(op_dir+o for o in operators_list)
    cmd = "trios_build combine %s %s %s" %(ops, imgset, fname)
    cmd = shlex.split(cmd)
    process = subprocess.call(cmd)
    if process != 0:
     raise Exception('Operation Failed')
    return process

def main(numwindows, npixels, winshape, level1trainset, level2trainset, testset, savetodir):
    winshape = winshape.split("x")
    height = int(winshape[0])
    width = int(winshape[1])
    make_windows(savetodir, (height,width), npixels, numwindows)

    #this is where i run the single level operators training
    for i in range(numwindows):
        print
        print "...Building operator %s ...\n" %str(i)
        build_operator(savetodir+"window_"+str(i)+".win", level1trainset, savetodir+"window_"+str(i)+"_op")

    #this is where i combine the single level operators to create a second level one
    for i in range(1,numwindows):
        print
        print "...Building operators combination : 0 to %s ... \n" %str(i)
        build_operators_combination(level2trainset,  np.array(range(i+1)), savetodir, savetodir+"twoLevel_0_to_"+str(i))

    triostestset = imageset.Imageset()
    testimgset = triostestset.read(testset)
    img_list = _get_imgList(testimgset)

    #applying the first level operators on the test set images
    for i in range(numwindows):
        for j in img_list:
            print
            print "...Applying operator %s on image : %s ...\n" %(str(i),j)
            ensemble.apply_operator(savetodir+"window_"+str(i)+"_op", j, savetodir+"window_"+str(i)+"_op-files/"+j.split("/")[-1][:-3]+"proc.pnm")
        ensemble.trios_test(savetodir+"window_"+str(i)+"_op", testset, savetodir+"window_"+str(i)+"_op-files/MAE.txt")

    #applying the second level operators combinations on the test set images
    for i in range(1,int(numwindows)):
        for j in img_list:
            ensemble.apply_operator(savetodir+"twoLevel_0_to_"+str(i),j, savetodir+"twoLevel_0_to_"+str(i)+"-files/level1/operator0/"+j.split("/")[-1][:-3]+"proc.pnm")
        #running trios_test tool to calculate MAE for current operator combination
        ensemble.trios_test(savetodir+"twoLevel_0_to_"+str(i), testset, savetodir+"twoLevel_0_to_"+str(i)+"-files/level1/operator0/MAE.txt")

    return 0

if __name__ == "__main__":
     parser = argparse.ArgumentParser(description="Performs experiment for randomly selected pixels.")
     parser.add_argument("-m", "--numwindows", type=int, help="Number of windows.")
     parser.add_argument("-n", "--npixels", type=int, help="The number of pixels from a window.")
     parser.add_argument("-w", "--winshape", help="Window shape, i.e. 11x11")
     parser.add_argument("-l1", "--level1trainset", help="Level 1 training set file path.")
     parser.add_argument("-l2", "--level2trainset", help="Level 2 training set file path.")
     parser.add_argument("-t", "--testset", help="Test set file path.")
     parser.add_argument("-s", "--savetodir", help="Directory to save files.")
     args = parser.parse_args()

     if not args.numwindows:
        print "Provide the number of windows to be built. -m --numwindows"
        raise Exception('Missing argument')
     if args.numwindows < 2:
        print "The number of wndows must be at least 2 ."
        raise Exception('Invalid number of windows.')
     if not args.npixels:
        print "Provide the number of pixels to be selected. -n --npixels"
        raise Exception('Missing argument')
     if not args.winshape:
        print "Provide the window shape (i.e 11,11 == 11x11 ). -w --winshape"
        raise Exception('Missing argument')
     if not args.level1trainset:
        print "Provide the train set on which the first level operators will be trained. -l1 --level1trainset"
        raise Exception('Missing argument')
     if not args.level2trainset:
        print "Provide the train set on which the seconde level operator will be trained. -l2 --level2trainset"
        raise Exception('Missing argument')
     if not args.testset:
        print "Provide the test set on which the operator will be applied. -t --testset"
        raise Exception('Missing argument')
     if not args.savetodir:
        print "Provide the directory to save the files. -s --savetodir"
        raise Exception('Missing argument')

     main(args.numwindows, args.npixels, args.winshape, args.level1trainset, args.level2trainset, args.testset, args.savetodir)