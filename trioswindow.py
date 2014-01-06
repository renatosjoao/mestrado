from matplotlib import pylab as pl
from PIL import Image, ImageDraw
from numpy import zeros, ones, array
from scipy.misc import toimage
import numpy

__all__ = ["plotpixels", "plotpixels_seq", "towindowfile", "savewindowfiles",
           "saveimagefiles", "invertpixelsnumbers"]

_WIN_HEADER = \
    "WINSPEC ########################################################"

def _assert_winshape(winshape):
    emsg1 = "Invalid window shape: %s" % (winshape,)
    if not len(winshape) == 2:
        raise ValueError, emsg1
    h, w = winshape
    if type(h) is int and type(h) is int:
        if h < 1 or w < 1:
            raise ValueError, emsg1
    else:
        raise ValueError, emsg1

def _assert_pixels(pixels, winshape):
    maxidx = winshape[0]*winshape[1] - 1
    if max(pixels) > maxidx:
        raise ValueError("Invalid pixels: %s, shape: %s" \
                          % (pixels, winshape))


def plot_pixels(pixels, winshape):
    """
    Plots one window using matplotlib.

    Parameters
    --------------------------------

    pixels: array like of n elements.
       The indices of pixels that belong to the window.

    winshape: two-element sequence.
        The shape of the window: (height, width)

    Returns
    --------------------------------

    window: array like of shape (height, width)
         The window, as a binary rectangular array.


    >>> w = plot_pixels([0, 2, 6, 7, 12, 13, 14, 33], (5, 7))

    """
    _assert_winshape(winshape)
    _assert_pixels(pixels, winshape)
    winlen = winshape[0]*winshape[1]
    window = zeros(winlen, numpy.int16)
    i1 = array(pixels, numpy.int32)
    window[i1] = 255
    window = window.reshape((winshape))
    pl.imshow(window, interpolation='nearest')
    pl.show()
    return window

def plot_pixels_seq(pixels_seq, winshape):
    """
    Convenience function to plot several windows at the same time.
    """
    for el in pixels_seq:
        plot_pixels(el, winshape)


def invert_pixels_numbers(pixels_seq, winshape):
    """
    Legacy function that inverts the indices of pixels relative to raster order.
    """
    _assert_winshape(winshape)
    maxidx = winshape[0]*winshape[1] -1
    def invert(cl):
        _assert_pixels(cl, winshape)
        aux = [(maxidx - el) for el in cl]
        return tuple(aux)
    result = [invert(el) for el in pixels_seq]
    return result


def to_window_file(pixels, winshape, winfile):
    """
    Creates a TRIOS window file for a given window.

    Parameters
    --------------------------------

    pixels: array like of n elements.
       The indices of pixels that belong to the window.

    winshape: two-element sequence.
        The shape of the window: (height, width)

    winfile: string.
        The name of the output file.


    Returns
    ---------------------------------

    None

    """
    _assert_winshape(winshape)
    _assert_pixels(pixels, winshape)
    winlen = winshape[0]*winshape[1]
    winh = int(winshape[0])
    winw = int(winshape[1])
    fd = open(winfile, "w")
    fd.write(_WIN_HEADER + "\n")
    fd.write(".h %d\n" % (winh,) )
    fd.write(".w %d\n" % (winw,) )
    fd.write(".d\n")
    c = zeros(winlen, numpy.int16)
    idx = array(pixels, numpy.int32)
    c[idx] = 1
    c = c.reshape((winshape))
    for row in range(winh):
        fd.write(" ".join([str(el) for el in c[row, :]]))
        fd.write("\n")
    fd.close()


def save_window_files(pixels_seq, winshape, prefix, suffix = "win"):
    """
    Convenience function to save several window files at the same time.

    Parameters
    --------------------------------

    pixels_seq: sequence of array like elements.
       A sequence of elements representing multiple windows. Each element of the sequence should be one object
        that could be passed on to 'to_window_file', i.e. an array like object containing indices of pixels.
        that belong to one given window.

    winshape: two-element sequence.
        The shape of the window: (height, width)

    prefix: string.
        Prefix to be prepended to the name of each output file.

    suffix: string
        Suffix to be appended to each output file.

    Returns
    ---------------------------------

    None
    """
    _assert_winshape(winshape)
    for pixels in pixels_seq:
        _assert_pixels(pixels, winshape)
    npixels_seq = len(pixels_seq)
    if npixels_seq < 10:
        fill = 1
    elif npixels_seq < 100:
        fill = 2
    else:
        fill = 3
    def strfill(number):
        strn = str(number)
        return (fill - len(strn))*"0" + strn
    for idx, pixels in enumerate(pixels_seq):
        fname = "%s%s%s%s" % (prefix, strfill(1+idx), ".", suffix)
        to_window_file(pixels, winshape, fname)

def to_image_file(pixels, winshape, winfile, scale=8):
    """
    Creates an image file for a given window.

    Parameters
    --------------------------------

    pixels: array like of n elements.
       The indices of pixels that belong to the window.

    winshape: two-element sequence.
        The shape of the window: (height, width)

    winfile: string.
        The name of the output file.

    scale: integer
        A positive integer, it is used as a multiplicative factor to determine the size of the output image.


    Returns
    ---------------------------------

    None

    """
    _assert_winshape(winshape)
    _assert_pixels(pixels, winshape)
    winlen = winshape[0]*winshape[1]
    winh = int(winshape[0])
    winw = int(winshape[1])
    c = 255*ones(winlen, numpy.uint8)
    idx = array(pixels, numpy.int32)
    c[idx] = 0
    c = c.reshape((winshape))
    imgc = toimage(c, mode="L")

    imgw, imgh = scale*winw + 1, scale*winh + 1
    imgc = imgc.resize((imgw, imgh), Image.NEAREST)
    draw = ImageDraw.Draw(imgc)
    for i in range(0, winh+1):
        # horizontal line
        draw.line([(0, scale*i), (imgw-1, scale*i)], fill="#000000")
    for i in range(0, winw+1):
        # vertical line
        draw.line([(scale*i, 0), (scale*i, imgh-1)], fill="#000000")
    imgc.save(winfile)
    return imgc

def save_image_files(pixels_seq, winshape, prefix, suffix = "eps"):
    """
    Convenience function to save several windows as images at the same time.

    Parameters
    --------------------------------

    pixels_seq: sequence of array like elements.
       A sequence of elements representing multiple windows. Each element of the sequence should be one object
        that could be passed on to 'to_image_file', i.e. an array like object containing indices of pixels.
        that belong to one given window.

    winshape: two-element sequence.
        The shape of the window: (height, width)

    prefix: string.
        Prefix to be prepended to the name of each output file.

    suffix: string
        Suffix to be appended to each output file.

    Returns
    ---------------------------------

    None
    """
    _assert_winshape(winshape)
    for pixels in pixels_seq:
        _assert_pixels(pixels, winshape)
    npixels_seq = len(pixels_seq)
    if npixels_seq < 10:
        fill = 1
    elif npixels_seq < 100:
        fill = 2
    else:
        fill = 3
    def strfill(number):
        strn = str(number)
        return (fill - len(strn))*"0" + strn
    for idx, pixels in enumerate(pixels_seq):
        fname = "%s%s%s%s" % (prefix, strfill(1+idx), ".", suffix)
        to_imagefile(pixels, winshape, fname)

