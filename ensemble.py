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
import xplutil
import classifier as clf
import numpy as np
import trioswindow as trios
from pylab import *

class Ensemble(object):

    def __init__(self, xpl_data, win, n_features, n_iterations, error_list, mae_list, dirpath):
        self.xpl_data = xpl_data
        self.win = win
        self.n_features = n_features
        self.n_iterations = n_iterations
        self.error_list = error_list
        self.mae_list = mae_list
        self.dirpath = dirpath

def train(xpl_data, n_features, n_iterations, dirpath):
    Xdata = xpl_data.data
    win = xpl_data.windata
    # copies frequency data as original frequencies are used towards the end to estimate training error
    w0 = xpl_data.freq0.copy()
    w1 = xpl_data.freq1.copy()
    error_list = []
    mae_list = []
    GVector = []
    DEC = np.zeros(w0.shape)

    w0_train, w1_train = clf.normalize_table(w0, w1)
    file = open(dirpath+"Error.txt", "w")

    for i in range(n_iterations):
        indices, feature_list, _ = ft.cmim(Xdata, w0, w1, n_features)
        trios.to_window_file(indices, xpl_data.winshape, dirpath+"window_"+str(i)+".win")
        trios.to_image_file(indices,xpl_data.winshape, dirpath+"window_"+str(i)+".png", scale=8)
        w0, w1 = clf.normalize_table(w0, w1)
        w0, w1, updated_decision, cls_error =  clf.apply_feature_selection(Xdata, indices, w0, w1)
        unique_array, unique_index = clf._apply_projection(Xdata, indices)
        clf.write_minterm_File(dirpath+"mtm"+str(i),indices, xpl_data.winshape, unique_array,updated_decision[unique_index])
        #str_to_file = "Classification error for iteration " + str(i) +" = "+ str(cls_error) +".\n"
        #file.write(str_to_file)
        error_list.append(cls_error)
        bt = clf.beta_factor(cls_error)
        gam = np.log(1/bt)
        GVector = np.append(GVector,gam)
        #DEC represents the Decision Table. Each column represents the decision for an iteration
        DEC = np.column_stack((DEC,updated_decision))
        aux_dec = DEC
        aux_dec = np.delete(aux_dec,0, axis=1)
        hypothesis = clf.adaboost_decision(aux_dec, GVector)
        MAE_t = clf.mae_from_distribution(hypothesis,w0_train, w1_train)
        mae_list = np.append(mae_list,MAE_t)
        str_to_file = "MAE for iteration " + str(i) +" = "+ str(MAE_t) +".\n\n"
        file.write(str_to_file)
    #Must delete the first column because it contains only Zeros as it was initialized with np.zeros()
    DEC = np.delete(DEC,0, axis=1)
    hypothesis = clf.adaboost_decision(DEC, GVector)
    MAE = clf.mae_from_distribution(hypothesis,w0_train, w1_train)
    str_to_file = "Final MAE = "+str(MAE)
    file.write(str_to_file)
    file.close()
    return Ensemble(xpl_data, win, n_features, n_iterations, error_list, mae_list,dirpath)


def min_empirical_error(xpldata):
    """
    Given the data originally from a XPL file the minimal empirical error
    is a value threshold for overfitting reference.

    Parameters
    ----------
    xpldata : ExampleData(data, freq0, freq1, winshape, windata, filename)
            Same as xplutil returns.

    Returns
    -------
    err : double
        The error value.
    """
    w0, w1 = clf.normalize_table(xpldata.freq0, xpldata.freq1)
    err  = clf.error(w0,w1)
    return err

def plot_MAE(xaxis, yaxis):
    """ This is a function to plot the MAE per iteration graph.

    Parameters
    ----------
        xaxis : array-like of shape = [n, 1]
            Iterations

        yaxis : array-like of shape = [n, 1]
            MAEs
        """

    plot(xaxis,yaxis)
    xlabel('Iteration (t)')
    ylabel('MAE')
    title('MAE per iteration')
    grid(True)
    show()

def predict(self, Xdata):
    return 0 #( TO BE IMPLEMENTED)
