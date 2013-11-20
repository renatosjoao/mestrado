##########################################
#Renato Stoffalette Joao
#
##########################################

__author__ = "Renato Stoffalette Joao(renatosjoao@gmail.com)"
__version__ = "$Revision: 0.1 $"
__date__ = "$Date: 2013// $"
__copyright__ = "Copyright (c) 2013 Renato SJ"
__license__ = "Python"


"""
Utility function to binarize image using OTSU threshold

 :Input:
     `image` : 

 Result: Returns a threshold value and the binarized Black & White image
     >>> return [thresh, im_bw]

"""

import cv2
        

def binarize(image):
    img_gray = cv2.imread(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    (thresh, im_bw) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return [thresh, im_bw]