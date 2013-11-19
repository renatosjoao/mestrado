#!/usr/bin/python
# -*- coding: latin-1 -*-

# Author: Carlos Santos

"""
Utility functions for dealing with xpl files.

:Function:

"""

__docformat__ = 'reStructuredText'

__all__ = ["read_xpl", "matrix_from_xpl", "parse_xpl "]

import re
import math
import numpy

def _debug(msg):
    print msg

# hexadecimal string regular expression
_hexre = re.compile('[0-9a-f]+', re.IGNORECASE)

# window data line regular expression
_windatare = re.compile('[0-9]+\s?')

_xpl_err_msg = "%s\nIn '%s', line n. %s"

_no_fill_bin_hex_dict = {'0':'', '1':'1', '2':'10', '3':'11',
    '4':'100', '5':'101', '6':'110', '7':'111',
    '8':'1000', '9':'1001', 'a':'1010', 'b':'1011',
    'c':'1100', 'd':'1101', 'e':'1110', 'f':'1111',
    'A':'1010', 'B':'1011', 'C':'1100', 'D':'1101',
    'E':'1110', 'F':'1111'}


_bin_hex_dict = {'0':'0000', '1':'0001', '2':'0010', '3':'0011',
    '4':'0100', '5':'0101', '6':'0110', '7':'0111',
    '8':'1000', '9':'1001', 'a':'1010', 'b':'1011',
    'c':'1100', 'd':'1101', 'e':'1110', 'f':'1111',
    'A':'1010', 'B':'1011', 'C':'1100', 'D':'1101',
    'E':'1110', 'F':'1111'}

# The correspondence among numeric codes and types was copied from
# `pac_kernel.h`. We should take care to keep it in sync.
_type_dict = {0:('BB', 'Binary x Binary'), 1:('BG', 'Binary x Grayscale'),
     2:('GB', 'Grayscale x Binary'), 3:('GG', 'Grayscale x Grayscale'),
     4:('WKF', 'Limited Grayscale x Limited Grayscale')}


def matrix_from_xpl(xpldata, dtype=numpy.uint8):
    """Turns xpl data into numpy arrays.

    The data conversion reverts the order of the xpl elements, so the returned
    numpy array matches the most commonly expected ordering.

    >>> xpldata = [('0000', {1: 7}), ('0001', {0: 3, 1: 2})]
    >>> data, freq0, freq1 = matrix_from_xpl(xpldata)
    >>> data[0], freq0[0], freq1[0]
    (array([0, 0, 0, 0], dtype=uint8), 0, 7)
    >>> data[1], freq0[1], freq1[1]
    (array([1, 0, 0, 0], dtype=uint8), 3, 2)

    """
    def to_numlist(bstring):
        result = [int(char) for char in bstring]
        result.reverse()
        return result
    def frequency(key, dict):
        if key in dict.keys():
            return dict[key]
        else:
            return 0
    def to_freqlist(key, xpl):
        return [frequency(key, el[1]) for el in xpldata]
    data = numpy.array([to_numlist(el[0]) for el in xpldata], dtype=dtype)
    freq0 = numpy.array(to_freqlist(0, xpldata))
    freq1 = numpy.array(to_freqlist(1, xpldata))
    return data, freq0, freq1


def bin_from_hex(hex_string, fill=None):
    """Returns a binary string (composed of only ones and zeros),
    corresponding to a hexadecimal string passed in as input.
    :Input:
        `hex_string`: hexadecimal string ([0-9a-f]+).
        `fill`: if `fill` is given, the resulting string will be left-filled
                with zeros so its length equals the value of `fill`.
    Return: a binary string.
    Raise: ValueError, if hex_string is not a valid hexadecimal string.

    >>> bin_from_hex('0xfb2')
    '111110110010'
    >>> bin_from_hex('0xfb2',16)
    '0000111110110010'
    >>> bin_from_hex('1248')
    '1001001001000'
    >>> bin_from_hex('1248', 16)
    '0001001001001000'

    """
    hex_string = hex_string.strip()
    if hex_string.startswith('0x'):
        hex_string = hex_string[2:]
    match = _hexre.match(hex_string)
    invalid_msg = "Invalid hexadecimal string: %s"
    if not match:
        raise ValueError(invalid_msg % (hex_string,))
    match_len = match.end() - match.start()
    if match_len != len(hex_string):
        raise ValueError(invalid_msg % (hex_string,))
    # no zero filling for most significant digit
    head = _no_fill_bin_hex_dict[hex_string[0]]
    tail = "".join([_bin_hex_dict[char] for char in hex_string[1:]])
    result =  head + tail
    if fill is not None:
        fill = int(fill)
        result_len = len(result)
        if fill < result_len:
            _debug("Could not fill result string!\n %s" % (result,))
        else:
            result = "0"*(fill - result_len) + result
    return result


def parse_tag_number(line, tag, linenumber, filename):
    """Parses a line which is composed of a tag followed by a numeric value
    and returns the numeric value.

    :Input:
    `line`: to be parsed (a string).
    `tag`: string containing the expected tag.
    `linenumber`: number of the line in the xpl file.
    `filename`: name of the xpl file.

    Return: the numeric values associated with the tag.

    Raises: `ValueError`, in case the line is malformed.

    >>> parse_tag_number('.t 2', '.t', 1, 'example.xpl')
    2

    """
    tokens = line.rstrip().split()
    if len(tokens) != 2:
        raise ValueError(_xpl_err_msg % ('Malformed line', filename,
                                          linenumber))
    # strips trailing whitespace, then splits line
    datatag, val = tokens
    if not tag == datatag:
        cause = "Invalid tag in: '%s'\nExpected: %s\n" % (line, tag)
        raise ValueError(_xpl_err_msg % (cause, filename, linenumber))
    try:
        intval = int(val)
        return intval
    except ValueError:
        cause = "Expected a number, got: '%s'" % (val)
        raise ValueError(_xpl_err_msg % (cause, filename, linenumber))


def read_win_data(lines, datastart, winsize, filename):
    """Reads window data.

    :Input:
    `lines`: sequence (tipically a list) of lines from the input file.
    `datastart`: index of the line which contains the `.d` tag indicating the
                 beginning of window data.
    `winsize`: tuples describing the window size (height, width).
    `filename`: name of xpl file.

    Return: a list of strings representing the window data; each string
            corresponds to one window row.

    """
    (height, width) = winsize
    result = []
    for linenum in range(datastart+1, datastart+1+height):
        data = lines[linenum]
        matches = _windatare.findall(data)
        if len(matches) == width:
            result.append(data.rstrip())
        else:
            cause = 'Malformed window data line:\n%s\n', (data)
            raise ValueError(_xpl_err_msg % (cause, filename, linenum))
    return result


def parse_header(lines, filename):
    type_val = parse_tag_number(lines[1], '.t', 1, filename)
    if not _type_dict.has_key(type_val):
        cause = "Unknown map type: '%s'" % (type_val)
        raise ValueError(_xpl_err_msg % (cause, filename, 1))
    nodes = parse_tag_number(lines[2], '.n', 2, filename)
    xpl_sum = parse_tag_number(lines[3], '.s', 3, filename)
    if nodes > xpl_sum:
        raise ValueError('Number of different examples is bigger \
                          than sum of examples')
    wintag = lines[4].rstrip() # strip trailing whitespace
    if wintag != ".W":
        cause = 'Invalid window tag %s: ' % (wintag,)
        raise ValueError(_xpl_err_msg % (cause, filename, 4))
    winheight = parse_tag_number(lines[5], '.h', 5, filename)
    winwidth = parse_tag_number(lines[6], '.w', 6, filename)
    if lines[7].startswith('.k'):
        datastart = 8
        wkrange = parse_tag_number(lines[7], '.k', 7, filename)
    else:
        datastart = 7
    windatatag =  lines[datastart].rstrip() # strip trailing whitespace
    if windatatag != ".d":
        cause = 'Invalid window data tag: %s' % (windatatag,)
        raise ValueError(_xpl_err_msg % (cause, filename, datastart))
    windata = read_win_data(lines, datastart, (winheight, winwidth),
                            filename)
    # increment so it points now to beginning of examples section
    datastart = datastart + winheight + 1
    result = {'windata':windata,
              'winsize':(winheight, winwidth),
              'maptype':_type_dict[type_val][0],
              'nodes':nodes,
              'sum':xpl_sum
             }
    return result, datastart


def parse_examplesBB(xpl_lines, winshape):
    result = []
    winlen = winshape[0]*winshape[1]
    # number of columns used for representation of each example
    # considering that each column can represent up to 32 bits
    xplcols = int(math.ceil(float(winlen)/32))

    # number of hexadecimal digits needed to represent the
    # biggest number which can be indexed through a window
    # of a given size
    hexdigits = int(math.ceil(float(winlen)/4))

    for dataline in xpl_lines:
        elemlist = dataline.strip().split()
        if elemlist[-1] != "0":
            cause = "Error: line must end with a zero:\n    "
            raise ValueError(cause + dataline)
        nelem = len(elemlist)
        assert (nelem == xplcols + 3) or (nelem == xplcols + 5), \
               "Error: wrong number of columns in line:\n    " + \
               dataline
        xpldigits = elemlist[0:xplcols]
        # List has to be reversed because most significant digit is
        # at rightnost column
        xpldigits.reverse()
        hexstring = "".join(xpldigits)
        if winlen < len(hexstring):
            emsg = "Error:\nString: %s\nColumns: %d" % (hexstring, xplcols)
            emsg += "Digits:\n" + "\n".join(xpldigits)
            raise ValueError, emsg
        datapoint = bin_from_hex(hexstring, fill=winlen)
        freqlist = elemlist[xplcols:nelem-1]
#         print elemlist
#         print xplcols
#         print freqlist
        freqdict = {}
        for i in range( len(freqlist)/2 ):
            freqdict[int(freqlist[2*i+1])] = int(freqlist[2*i])
        result.append((datapoint, freqdict))
    return result


def parse_lines(lines, data_id="XPL data"):
    """Parses the content of xpl file, represented as a sequence of lines.

    lines: file data as a sequence of lines.
    data_id: data identifier for debugging purposes, tipically the file name.

    >>> ll = ["EXAMPLE ########################################################"]
    >>> ll.append(".t 0")
    >>> ll.append(".n 2")
    >>> ll.append(".s 17")
    >>> ll.append(".W")
    >>> ll.append(".h 2")
    >>> ll.append(".w 2")
    >>> ll.append(".d")
    >>> ll.append("1 1")
    >>> ll.append("1 1")
    >>> ll.append(".d")
    >>> ll.append("0 7 1 0")
    >>> ll.append("1 3 0 2 1 0")
    >>> file_content = parse_lines(ll)
    >>> xpldata = file_content['xpldata']
    >>> xpldata[1][0]
    '0001'
    >>> xpldata[1][1][0]
    3
    >>> xpldata[1][1][1]
    2
    >>> xpldata[0][0]
    '0000'
    >>> xpldata[0][1][1]
    7

    """
    result, datastart = parse_header(lines, data_id)
    winheight, winwidth = result['winsize']
    datatag = lines[datastart].rstrip() # strip trailing whitespace
    if datatag != ".d":
        cause = 'Invalid data tag: ' % (datatag,)
        raise ValueError(_xpl_err_msg % (cause, data_id, datastart))
    maptype = result['maptype']
    if maptype == 'BB':
        xpldata = parse_examplesBB(lines[(datastart+1):],\
                                    (winheight, winwidth))
    else:
        cause = "Parsing of '%s' examples is not implemented"
        raise ValueError(cause % (maptype,))
    result['xpldata'] = xpldata
    return result


def parse_xpl(filename):
    """
    Parses a xpl file.

    :Input:
    `filename`: xpl file name.
    """
    infile = None
    try:
        infile = open(filename)
        # TODO> I believe the call to readlines is unnecessary here
        lines = infile.readlines()
        result = parse_lines(lines, filename)
        return result
    finally:
        if infile is not None:
            infile.close()

def matrix_from_windata(windata):
    data_tokens = [line_data.strip().split() for line_data in windata]
    data = [[int(elem) for elem in tokens] for tokens in data_tokens]
    return numpy.array(data, dtype=numpy.uint8)


class ExampleData(object):
    """Describes an example set."""
    def __init__(self, data, freq0, freq1, winshape, windata=None, tag=None):
        self.data = data
        self.freq0 = freq0
        self.freq1 = freq1
        self.winshape = winshape
        self.windata = windata
        self.tag = tag


def read_xpl(filename):
    # Soon after reading, the strings contained in xplcontent are ordered
    # such that the most significant digit of the string corresponds to
    # the lower right corner of the window. For instance, the string
    # '0011' translates into the 2x2 window:
    #  1 1
    #  0 0
    xplcontent = parse_xpl(filename)
    # when converting to a numpy array, the ordering of the elements
    # is fixed
    data, freq0, freq1 = matrix_from_xpl(xplcontent['xpldata'])
    windata = matrix_from_windata(xplcontent['windata'])
    winshape = xplcontent['winsize']
    result = ExampleData(data, freq0, freq1, winshape, windata, filename)
    return result


def _test():
    import doctest
    doctest.testmod()


if __name__ == "__main__":
    _test()
