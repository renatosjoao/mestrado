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
import os.path
import classifier as clf
import feature as ft
import xplutil
import numpy as np
import trioswindow as triosw
import pylab
import matplotlib.pyplot as plt
import subprocess
import shlex
from scipy import misc

class Ensemble:

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
    total = float(np.sum([w0, w1]))
    w0_train = w0/total
    w1_train = w1/total

    file = open(dirpath+"MAE.txt", "w")

    for i in range(n_iterations):
        indices, feature_list, _ = ft.cmim(Xdata, w0, w1, n_features)
        triosw.to_window_file(indices, xpl_data.winshape, dirpath+"window_"+str(i)+".win")
        triosw.to_image_file(indices,xpl_data.winshape, dirpath+"window_"+str(i)+".png", scale=8)
        total = float(np.sum([w0, w1]))
        w0 = w0/total
        w1 = w1/total
        w0, w1, updated_decision, cls_error =  clf.apply_feature_selection(Xdata, indices, w0, w1)
        unique_array, unique_index = clf._apply_projection(Xdata, indices)
        xplutil.write_minterm_file(dirpath+"mtm"+str(i),indices, xpl_data.winshape, unique_array,updated_decision[unique_index])
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
        str_to_file =  str(i) +", "+ str(MAE_t) +"\n"
        file.write(str_to_file)
    #Must delete the first column because it contains only Zeros as it was initialized with np.zeros()
    DEC = np.delete(DEC,0, axis=1)
    hypothesis = clf.adaboost_decision(DEC, GVector)
    #MAE = clf.mae_from_distribution(hypothesis,w0_train, w1_train)
    #str_to_file = "Final MAE = "+str(MAE)
    #file.write(str(MAE))
    file.close()
    plot_MAE(np.array(range(n_iterations)), np.array(mae_list), dirpath)
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

def min_mae(imageset_list):
    """
    Given the observed image data  and the ideal image data we calculate the
    mae as improvement threshold

    Parameters
    ----------
    imageset_list: list
        A list containing the imageset to be used for mae calculation.
    i.e.
    [('./dataset-map/map1bin.pnm', './dataset-map/map1bin.ide.pnm', './dataset-map/map1bin.pnm'),
    ('./dataset-map/map2bin.pnm', './dataset-map/map2bin.ide.pnm', './dataset-map/map2bin.pnm'),
    ('./dataset-map/map3bin.pnm', './dataset-map/map3bin.ide.pnm', './dataset-map/map3bin.pnm')]

    Returns
    -------
    value : double

    """
    sum_nonzero = 0.0
    total_pix = 0.0
    for row in imageset_list:
        ideal = misc.imread(row[1])
        observed = misc.imread(row[0])
        subset = np.absolute(ideal - observed)
        nonzero = np.count_nonzero(subset)
        sum_nonzero += nonzero
        total_pix += subset.size
    value = sum_nonzero/total_pix
    return value

def plot_MAE(xaxis, yaxis, dir):
    """ This is a function to plot the MAE per iteration graph.

    Parameters
    ----------
        xaxis : array-like of shape = [n, 1]
            Iterations

        yaxis : array-like of shape = [n, 1]
            MAEs
        """
    fig = plt.figure()
    plt.title('MAE per iteration')
    plt.xlabel('Iteration (t)')
    plt.ylabel('MAE')
    plt.grid(True)
    plt.plot(xaxis, yaxis)
    fig.savefig(dir +'MAE_trainning.png', dpi=fig.dpi)

def predict(self, Xdata):
    return 0 #( TO BE IMPLEMENTED)

def build_xpl(imgset, win, output):
    """
    Writes a xpl file to disk according to the parameters

    Parameters
    ----------
    imgset : string
            The image set path for creatng the XPL file.
    win : string
            The path to the win file used to create the XPL file.
    output : string
            The path to the output XPL file.

    Returns:
    --------
            : bolean value
            True if the processing is sucessful
    """
    #trios_build_xpl must not be hardcoded !!!
    cmd = "/home/rsjoao/git/trioslib-code/bin/trios_build_xpl %s %s %s" %(imgset, win, output)
    cmd = shlex.split(cmd)
    process = subprocess.call(cmd)
    if process == 0:
        return True
    else:
        raise Exception('Build XPL operation failed')

def build_operators(dirpath, n_iterations):
    for i in range(n_iterations):
        window = dirpath+"window_"+str(i)+".win"
        mtm = dirpath+"mtm"+str(i)
        output = dirpath+"mtm"+str(i)+"-op"
        build_operator(window, mtm, output)
    return 0

def build_operator(win, mtm, output):
    """
    Runs the build operator process using the mimtermfile passed in the parameters.

    Parameters
    ----------
    win : string
            The path to the win file used to create the image operator.
    mtm : string
            The path to the mimterm file used to create the image operator.
    output : string
            The path to the output image operator.
    """
    #trios_build_operator must not be hardcoded !!!
    cmd = "/home/rsjoao/git/trioslib-code/bin/trios_build_operator %s %s %s" %(win, mtm, output)
    cmd = shlex.split(cmd)
    process = subprocess.call(cmd)
    if process == 0:
        return True
    else:
        raise Exception('Build operator failed')

def build_operator_combination(imgset, op_to_combine, op_dir, fname):
    #op_to_combine i.e 1,2,3 or 2,3,4 or 3,4,5
    operators_list = []
    outputfname = fname
    for i in op_to_combine:
        operators_list.append("mtm"+str(i)+"-op")
    p = build_2level(imgset, operators_list, op_dir, outputfname)
    if p == True:
        return True
    else:
        raise Exception('Build operator failed')

def build_2level(imgset, operators, op_dir, fname):
    """
    This function builds the 2-level operator by running trios_build tool
    with the flag 'combine'.

    Parameters
    ----------
    imgset : string
            The image set path for training the 2-level operator.
    operators : string[]
            List of operators that will be combined to create the 2-level operator.
    op_dir : string
            The directory path related to where the operators are located.
    fname : string
            The 2-level image operator path after combining the first level operators.

    Returns:
    --------
            : bolean value
            True if the processing is sucessful
    """
    ops = " ".join(op_dir+o for o in operators)
    cmd = "/home/rsjoao/git/trioslib-code/bin/trios_build combine %s %s %s" %(ops, imgset, fname)
    cmd = shlex.split(cmd)
    process = subprocess.call(cmd)
    if process == 0:
        return True
    else:
        raise Exception('Building Image Operator failed')

def apply_operator(operator_path, img_path, result_path, mask=''):
    """
    Apply a trained operator on an image.

    Parameters
    ----------
    operator_path : string
            The operator path.
    img_path : string
            The image path on which the operator will be applied.
    result_path : string
            The image result path after applying the operator.
    mask: string
            The mask image path. Optional.

    Returns:
    --------
            0 if the processing is sucessful.
    """
    cmd = "/home/rsjoao/git/trioslib-code/bin/trios_apply %s %s %s %s"%(operator_path, img_path, result_path, mask)
    cmd = shlex.split(cmd)
    res = subprocess.call(cmd)
    if res != 0:
     raise Exception('Applying Image Operator Failed')
    return res