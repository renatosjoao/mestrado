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


import detect


def apply_operator(self, operator_path, img_path, result_path, mask=''):
    """
    Apply a trained operator on an image.
    img can be a path to a PGM image on the disk or a PIL Image.
    """
    res = detect.call('trios_apply %s %s %s %s'%(operator_path, img_path, result_path, mask))
    if res != 0:
     raise Exception('Apply failed')
    return res     
     


def build_operator(self, win, mtm, output):
    """
    Runs the build operator process using the mimtermfile passed in the parameters.
    """
    process = detect.call('trios_build_operator %s %s %s'%(win, mtm, output))
    if process == 0:
        self.built = True
    else:
        self.built = False
        raise Exception('Build operator failed')


def build_xpl(self, imgset, win, output):
    """
    Writes a xpl file to disk according to the parameters
    """
    process = detect.call('trios_build_xpl %s %s %s'%(imgset, win, output))
    if process == 0:
        self.built = True
    else:
        self.built = False
        raise Exception('build xpl operation failed')