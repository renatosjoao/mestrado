# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 12:11:07 2014

@author: rsjoao
"""

import os, sys
import argparse
thisdir = os.path.dirname(__file__)
libdir = os.path.join(thisdir, '../')

if libdir not in sys.path:
    sys.path.insert(0, libdir)

import subprocess
import shlex

def trios_test(operator_path, imgset_path, output):
    """
    Calculates the image operator MAE over a image set

    Parameters
    ----------
    operator_path : string
            The operator path.

    imgset_path : string
            The imageset path on which the operator will be applied.
    output : string
            The output file on which the MAE will be written to.
    """
    f = open(output, "w")
    f2 = open("/dev/null", "w")
    cmd = "trios_test %s %s" %(operator_path,imgset_path)
    cmd = shlex.split(cmd)
    res = subprocess.call(cmd,stdout=f, stderr=f2)
    if res != 0:
     raise Exception('Operation Failed')
    f.close()
    f2.close()
    return res


def main(testset, operator, output):

    trios_test(operator, testset, output)


if __name__ == "__main__":
     parser = argparse.ArgumentParser(description="Performs MAE calculation")
     parser.add_argument("-t", "--testset", help="Test set file path.")
     parser.add_argument("-o", "--operator", help="Operator file path.")
     parser.add_argument("-s", "--savetofile", help="Save to output file.")
     args = parser.parse_args()

     if not args.testset:
        print "Provide test set file. -t --testset"
        raise Exception('Missing argument')
     if not args.operator:
        print "Provide operator file path. -o --operator"
        raise Exception('Missing argument')
     if not args.savetofile:
        print "Provide the file name to save the output into it. -s --savetofile"
        raise Exception('Missing argument')

     main(args.testset, args.operator, args.savetofile)