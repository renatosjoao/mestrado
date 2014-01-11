# -*- coding: utf-8 -*-
##########################################
#Renato Stoffalette Joao
#
##########################################

__author__ = "Renato Stoffalette Joao(renatosjoao@gmail.com)"
__version__ = "$Revision: 0.1 $"
__date__ = "$Date: 2013// $"
__copyright__ = "Copyright (c) 2013 Renato SJ"
__license__ = "Python"

import numpy as np

def mean_absolute_error(true_values, pred_values):
    """Mean absolute error function

    Parameters
    ----------
    true_values : array-like of shape = [n_samples]
        Ground truth target values.

    pred_values : array-like of shape = [n_samples]
        Estimated target values.

    Returns
    -------
    mae : float
        A positive floating point value (the best value is 0.0).

    """
    return np.mean(np.abs(pred_values - true_values))