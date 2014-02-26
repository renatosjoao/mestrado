# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 13:36:52 2014

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

def project(table, i):
    hash = {}
    subset = table[:, i]
    unique_array, unique_index = cl._apply_projection(table, i)
    for row in unique_array:
        hash[tuple(row)] = np.where(np.all(row==subset,axis=1))
    return hash, unique_array


# a ideia aqui e passar uma janela com 1 em todos os indices
def main(trainset, window, save_todir):
    XPL = window+'.xpl'
    #building xpl file
    ensemble.build_xpl(trainset,window,XPL)
    #reading xpl file
    xpl_data = xplutil.read_xpl(XPL)
    indices = np.array([0,1,3,4,5,7,8])
    w0 = xpl_data.freq0.copy()
    w1 = xpl_data.freq1.copy()
    w0, w1 = cl.normalize_table(w0, w1)
    hash, unique_array = project(xpl_data.data, indices)
    sum0 = []
    sum1 = []
    for row in unique_array:
        arr = hash.get(tuple(row.reshape(1,-1)[0]))
        indexes =  tuple(arr[0].reshape(1,-1)[0])
        sum0.append(w0[[np.array(indexes)]].sum())
        sum1.append(w1[[np.array(indexes)]].sum())
    decision = cl.make_decision(sum0, sum1)
    xplutil.write_minterm_file(save_todir+"mtmFile.mtm",indices, xpl_data.winshape, unique_array,decision)

if __name__ == "__main__":
     parser = argparse.ArgumentParser(description="Create a MINTERM file given a training set file and a window file.")
     parser.add_argument("-t", "--trainset", help="Training set file path")
     parser.add_argument("-w", "--window", help="Window file path.")
     parser.add_argument("-s", "--savetodir", help="Directory to save files.")
     args = parser.parse_args()
     main(args.trainset, args.window, args.savetodir)