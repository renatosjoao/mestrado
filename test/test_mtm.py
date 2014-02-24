# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 17:23:36 2014

@author: rsjoao
"""
import os, sys
import argparse

thisdir = os.path.dirname(__file__)
libdir = os.path.join(thisdir, '../')

if libdir not in sys.path:
    sys.path.insert(0, libdir)

import numpy as np
import classifier as cl
import ensemble
import xplutil

def main(trainset, window, save_todir):
    XPL = save_todir+'file.xpl'
    #building xpl file
    ensemble.build_xpl(trainset,window,XPL)
    #reading the xpl file
    xpl_data = xplutil.read_xpl(XPL)
    decision = cl.make_decision(xpl_data.freq0,xpl_data.freq1)
    size = xpl_data.winshape[0]*xpl_data.winshape[1]
    xplutil.write_minterm_file(save_todir+'filename.mtm', np.array([range(size)]), xpl_data.winshape, xpl_data.data, decision)

if __name__ == "__main__":
     parser = argparse.ArgumentParser(description="Create a MINTERM file given a training set file and a window file.")
     parser.add_argument("-t", "--trainset", help="Training set file path")
     parser.add_argument("-w", "--window", help="Window file path.")
     parser.add_argument("-s", "--savetodir", help="Directory to save files.")
     args = parser.parse_args()
     main(args.trainset, args.window, args.savetodir)