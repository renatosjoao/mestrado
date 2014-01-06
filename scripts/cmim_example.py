# -*- coding: utf-8 -*-
import _mypath

import feature
import trioswindow
import xplutil

import argparse

def main(xpl_data, num_features=None, plotwindow=False, winfile=None):
    height, width = xpl_data.winshape
    indices, feature_list, _ = feature.cmim(xpl_data.data, xpl_data.freq0,
                                            xpl_data.freq1, num_features)
    print "Selected features: ", indices
    if plotwindow:                                        
        trioswindow.plot_pixels(indices, xpl_data.winshape)
    if winfile:
        trioswindow.to_window_file(indices, xpl_data.winshape, winfile)
    
if __name__ == "__main__":
     parser = argparse.ArgumentParser()
     description="Applies CMIM feature selection to a given XPL file."
     parser.add_argument("filename", help="XPL filename")
     parser.add_argument("-w", "--window",
              help="Trios window file name where to save the obtained window")
     parser.add_argument("-n", "--numfeatures", type=int,
                         help="Maximum number of features")
     parser.add_argument("-s", "--showwindow", action="store_true",
                         help="Show window obtained after feature selection.")      
     args = parser.parse_args()
     if args.filename:
         xpl_data = xplutil.read_xpl(args.filename)    
         main(xpl_data, args.numfeatures, args.showwindow, args.window)
     else:
         print "Must provide input file name."
        
        

