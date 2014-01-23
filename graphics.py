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
from pylab import *

def plot_MAE_iter(xaxis, yaxis):
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
