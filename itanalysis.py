#!/usr/bin/python
# -*- coding: latin-1 -*-

# Author: Carlos Santos

"""
:Main data structures:

    - 2D histogram: a numpy array (ndim = 2), shape = (2, 2). For a given pair
    of binary random variables, gives the histogram estimator of their joint
    probability.

    - Matrix of 2D histograms: A numpy array (ndim = 4), shape = (N-1, N, 2, 2)
    The first two indices denote a pair of elements in the original data vector
    The last two indices refer to values (0, 1) which the elements may take.

:Function:

"""

__docformat__ = 'reStructuredText'

__all__ = ["hist2dmatrix", "entropy", "makelocalpairgenerator", \
           "mixedpairgenerator", "mutual_info", "create_naivebayes", \
           "interactions", "attr_analysis", "sequential_apply_naive_bayes"]

import numpy
from   numpy import zeros, array, float32, int8, \
                    int16, int32, log2

from math    import log


# float limits
_flimits = numpy.finfo(numpy.float)

def _to_binary(x):
    if x:
        return 1
    else:
        return 0

def _array_from_dict(dict, shape,  dtype=int32):
    """Takes a dictionary representing a matrix as input and returns a
    numpy.ndarray object containing the same data.
    :Input:
    dict: a dictionary whose keys are tuples of non-negative integer numbers
          representing matrix indices and whose values are matrix entries.
    shape: shape of the numpy.ndarray object to be created
    dtype: type of the array to be created; defaults to numpy.int32

    >>> dict = {(0,0):1, (1,1):2}
    >>> _array_from_dict(dict, (2,2))
    array([[1, 0],
           [0, 2]])


    """
    keylist = dict.keys()
    assert len(keylist) > 0, "Empty input dictionary"
    result = numpy.zeros(shape, dtype)
    for key in dict:
        try:
            result[key] = dict[key]
        except:
            emsg = "Failed key: %s; Value: %s; Array shape: %s"
            raise ValueError, emsg % (key, dict[key], shape)
    return result

def _print_dict(dict):
    keylist = dict.keys()
    keylist.sort()
    for key in keylist:
        print "%s: %s" % (key, dict[key])


def _orderedtriples(x):
    """Produces all distinct triples (a,b,c) where 0 <= a < b < c < x
    and a, b, c, x are integers
    """
    for i in range(x-2):
        for j in range(i+1, x-1):
            for k in range(j+1, x):
                yield (i, j, k)

def _orderedpairs(x):
    """Produces all distinct pairs (a,b) where 0 <= a < b < x
    and a, b, x are integers
    """
    for i in range(x-1):
        for j in range(i+1, x):
                yield (i, j)


def entropy(p):
    """Calculates sum(-p_k*log2 p_k) where the values p_k compose a normalized
probability distribution: p_k >= 0 for all k; and sum(p_k) = 1.
    :Input:
    p: probability distribution (possibly not normalized)

    >>> entropy([1, 0])
    0.0
    >>> entropy([1, 1])
    1.0
    >>> entropy([1, 1, 1, 1])
    2.0
    >>> entropy([1, 1, 1, 1, 1, 1, 1, 1])
    3.0

    """
    parray = numpy.array(p, dtype=float32)
    if not (parray >= 0).all():
        raise ValueError, "Negative probability"
    parray = parray / parray.sum()
    # non-null entropy terms
    pset =  (parray > 0) & (parray < 1)
    return (-parray[pset]*log2(parray[pset])).sum()


def hist2dmatrix(data, freq, dtype=int32):
    """Calculates the collection of 2 dimensional histograms for a binary
dataset.
    :Input:
    data: a dataset passed in as a numpy 2d array.
    freq: an array fo frequencies (counts)
    dtype: Data type for the resulting array; defaults to numpy.int32
    :Output:
    An array containing the frequencies. The array keys have the format
(i, j, v_i, v_j), where (i,j) correspond to two different
components in the binary vector and (v_i, v_j) correspond to values
assumed by these components; thus (v_i, v_j) is a binary sequence. If the
original binary vector had length N, then the output array will have shape
(N-1, N, 2, 2)

    >>> import numpy
    >>> data = numpy.array([(0, 0, 0), (0, 0, 1), (0, 1, 0)])
    >>> freq = numpy.array([1, 1, 2])
    >>> h2d = hist2dmatrix(data, freq)
    >>> h2d[0,1,:,:]
    array([[2, 2],
           [0, 0]])
    >>> h2d[0,2,:,:]
    array([[3, 1],
           [0, 0]])
    >>> h2d[1,2,:,:]
    array([[1, 1],
           [2, 0]])

    """
    if data.ndim != 2:
        raise ValueError, "Input data array of wrong shape %s" % (data.shape,)
    npoints, veclen = data.shape
    if freq.shape[0] != npoints:
        raise ValueError, "Number of data points and number of frequency \
             entries do not match: %d data points; %d frequency entries" % \
             (npoints, freq.shape[0])

    result = zeros((veclen-1, veclen, 2, 2), dtype)

    for point in range(npoints):
        pointfreq = freq[point]
        for i in range(veclen-1):
            rowval = _to_binary(data[point, i])
            for j in range(i+1, veclen):
                colval = _to_binary(data[point, j])
                result[i, j, rowval, colval] += pointfreq
    return result



def hist2dto1d(hist2d, datalen):
    """This method transforms a set of 2d histograms into a set of  1d
    histograms by summing over one of the components.
    :Input:
    hist2d: a set of 2d histograms, like the one created
            with mhist3dto2d
    datalen: length of binary vector which originated the 2d histograms
    :Output:
    An array containing the frequencies. The array keys have the format
(i, v_i), where (i) correspond to one components in the binary vector
and (v_i) correspond to values assumed by this component; thus (v_i) is
a binary value. If the original binary vector had length N, then the output
array will have shape (N, 2)

    >>> import numpy
    >>> data = numpy.array([(0, 0, 0), (0, 0, 1), (0, 1, 0)])
    >>> freq = numpy.array([1, 1, 2])
    >>> h2d = hist2dmatrix(data, freq)
    >>> h1d = hist2dto1d(h2d, 3)
    >>> h1d
    array([[4, 0],
           [2, 2],
           [3, 1]])

    """
    if not hist2d.ndim == 4:
           raise ValueError, \
                 "2D histogram matrix has wrong size %s" % (hist2d.shape,)
    _, xlen, _, _ = hist2d.shape

    result = hist2d[0,:,:,:].sum(1)
    res2 = hist2d[0,1,:,:].sum(1)
    #print "result shape: ", result.shape
    #print "res2 shape: ", res2.shape
    #print "xlen: ", xlen
    result[0, :] = res2
    return result




def _info_gain(h2d, x, y, base=2):
    """Information gain for a putative transformation of a pair of variables.
    :Input:
    h2d: collection of 2d histograms
    x, y: pair of variables to be used in calculations
    base: base of logarithm to use for gain calculations, defaults to 2
    :Output:
    gain: information gain of transformation

    """
    assert x >= 0
    assert y > x
    assert h2d.ndim == 4
    freqs = h2d[x,y,:,:]
    # invariant part of the gain, not dependent
    # on whether X or Y is replaced
    g0 = entropy((freqs[0,0]+freqs[1,1], freqs[0,1]+freqs[1,0]))
    # Entropy of X
    hx = entropy(freqs.sum(1))
    # Entropy of Y
    hy = entropy(freqs.sum(0))
    gain = (hx - g0, hy - g0)
    return gain

def _info_gain_matrix(h2d, dlen, gtype=float32):
    """Conditional information gain matrix.
    :Input:
    h2d: 2 dimensional histogram
    dlen: data length.
    gtype: type of gain matrix, it defaults to numpy.float32
    :Output:
    gmat: Information gain matrix

    """
    # gain matrix
    gmat = zeros((dlen,dlen, 2), gtype) # float matrix

    for x,y in _orderedpairs(dlen):
        gmat[x,y, :] = _info_gain(h2d, x, y)
    return gmat

def _cluster_fusion(c1,c2):
    """Joins two clusters of components, eliminating the repeated elements
    between them.
    :Input:
    c1 : first cluster
    c3: second cluster
    :Output:
    Resulting cluster

    The clusters passed in as arguments must be ordered, e.g. (1,4,7,11).
    The resulting cluster will preserve the ordering.

    >>> _cluster_fusion((2,),(1,))
    (1, 2)
    >>> _cluster_fusion((1,4,6,8), (1,3,5,6,10))
    (3, 4, 5, 8, 10)

    """
    new_cluster = []
    pt1 = 0
    pt2 = 0
    while pt1 < len(c1) and pt2 < len(c2):
        if c1[pt1] < c2[pt2]:
            new_cluster.append(c1[pt1])
            pt1 += 1
        elif c1[pt1] > c2[pt2]:
            new_cluster.append(c2[pt2])
            pt2 += 1
        else:
            pt1 += 1
            pt2 += 1
    if pt1 < len(c1):
        new_cluster.extend(c1[pt1:])
    elif pt2 < len(c2):
        new_cluster.extend(c2[pt2:])
    return tuple(new_cluster)


def _cluster_union(c1, c2):
    """Performs the union (set operation) of two clusters of components.
    :Input:
    c1: first cluster
    c3: second cluster
    :Output:
    Union cluster

    The clusters passed in as arguments must be ordered, e.g. (1,4,7,11).
    The resulting cluster will preserve the ordering.

    >>> _cluster_union((2,),(1,))
    (1, 2)
    >>> _cluster_union((1,4,6), (1,3,5,6))
    (1, 3, 4, 5, 6)

    """
    union_list = list(c1) + [el for el in c2 if not el in c1]
    union_list.sort()
    return tuple(union_list)



def _matrix_from_dataset(dataset):
    """Convert dataset in format [(vector,freq)] to matrices.
    :Input:
    dataset

    :Return:
    (M, F)
    M: data matrix
    F: frequency matrix

    >>> dataset = (((0,0,0),1), ((0,0,1),3), ((0,1,1),5))
    >>> m,f = _matrix_from_dataset(dataset)
    >>> m
    array([[0, 0, 0],
           [0, 0, 1],
           [0, 1, 1]], dtype=int8)
    >>> f
    array([1, 3, 5])


    """
    M = numpy.array([el[0] for el in dataset], dtype=int8)
    F = numpy.array([el[1] for el in dataset], dtype=int32)
    return M, F


def _bin_entr(p, base=2):
    """Entropy of a binary variable.
    :Input:
    p: probability
    base: base of logarithm
    :Output:
    Entropy value

    >>> _bin_entr(0)
    0
    >>> _bin_entr(1)
    0
    >>> _bin_entr(0.5)
    1.0

    """

    assert p >= 0, "Negative probability"
    assert p <= 1, "Probability is greater than one."
    if p == 0 or p == 1:
        return 0
    else:
        return -p*log(p, base) - (1-p)*log(1-p, base)

def _vec_bin_entr(p):
    """Entropy of a vector of binary variables.
    :Input:
    p: probability
    base: base of logarithm
    :Output:
    Entropy value

    >>> _vec_bin_entr(numpy.array([0, 0.25, 0.5, 0.75, 1.0]))
    array([ 0.        ,  0.81127812,  1.        ,  0.81127812,  0.        ])


    """

    assert (p >= 0).all(), "Negative probability"
    assert (p <= 1).all(), "Probability is greater than one."
    pset = (p > 0) & (p < 1)
    result = zeros(p.shape, dtype=p.dtype)
    result[pset] = -p[pset]*log2(p[pset]) - (1-p[pset])*log2(1-p[pset])
    return result


def _histogram2d(data, freq):
    """Two dimensional histogram of binary data.
    :Input:
    data: a Nx2 binary data array
    freq: a Nx1 array of frequencies associated with the patterns in data.
    :Output:
    A two dimensional histogram

    >>> d = numpy.array([[0,0], [0,1], [1,0], [1,1], [0,1], [1,0]])
    >>> f = numpy.array([1, 2, 3, 5, 7, 11])
    >>> h = _histogram2d(d, f)
    >>> h
    array([[ 1,  9],
           [14,  5]])

    """
    if freq.ndim != 1:
           raise ValueError, \
                 "Wrong shape of frequency vector: %s" % (freq.shape,)
    npoints, dim = data.shape
    npointsf, = freq.shape
    if npoints != npointsf:
           raise ValueError, \
                "Number of examples in data and freq do not match."
    if dim != 2:
        raise ValueError, "Wrong shape of data vector: %s" % (data.shape,)

    result = zeros((2,2), dtype=freq.dtype)
    # The most obvious implementation with array indexing tricks did not
    # work, so I had to resort to the following one using a for loop
    for i in range(npoints):
        row = _to_binary(data[i,0])
        col = _to_binary(data[i,1])
        result[row, col] = result[row, col] + freq[i]
    return result


def conditional_entropy(p0, p1, weights=None):
    """Conditional entropy of a given variable conditioned on output
    variable.
    p0: distribution of input variable, given that output = 0
    p1: distribution of input variable, given that output = 1

    >>> from numpy import array
    >>> p0 = numpy.array([1, 0])
    >>> p1 = numpy.array([0, 1])
    >>> mutual_info(p0, p1)
    1.0
    >>> p0 = numpy.array([4, 2])
    >>> p1 = numpy.array([2, 1])
    >>> mutual_info(p0, p1)
    0.0
    >>> p0 = numpy.array([1, 0, 0, 0])
    >>> p1 = numpy.array([0, 1, 1, 1])
    >>> mutual_info(p0, p1)
    0.811278045177
    >>> p0 = numpy.array([4, 4, 0, 0])
    >>> p1 = numpy.array([1, 1, 2, 4])
    >>> mutual_info(p0, p1)
    0.548794865608

    """

    if weights is None:
        t0 = p0.sum()
        t1 = p1.sum()
    else:
        t0, t1 = weights
    total = t0 + t1

    # entropy of the input variable
    ent = entropy(p0 + p1)
    # entropy of the input variable, conditioned on output value
    cond_ent = (t0*entropy(p0) + t1*entropy(p1))/float(total)
    # Allows for small numerical errors that might make
    # cond_ent go negative
    if cond_ent < 0 and abs(cond_ent/ent) < 1e-6:
        cond_ent = 0
    if cond_ent < 0:
        emsg = "Conditional Entropy < 0!\nH(X) = %s\nH(X|Y)  = %s\n"
        raise ValueError, emsg % (ent, cond_ent,)

    return cond_ent



def mutual_info(p0, p1, weights=None):
    """Mutual information between the output variable and a second
    variable.
    p0: distribution of input variable, given that output = 0
    p1: distribution of input variable, given that output = 1

    >>> from numpy import array
    >>> p0 = numpy.array([1, 0])
    >>> p1 = numpy.array([0, 1])
    >>> mutual_info(p0, p1)
    1.0
    >>> p0 = numpy.array([4, 2])
    >>> p1 = numpy.array([2, 1])
    >>> mutual_info(p0, p1)
    0.0
    >>> p0 = numpy.array([1, 0, 0, 0])
    >>> p1 = numpy.array([0, 1, 1, 1])
    >>> mutual_info(p0, p1)
    0.811278045177
    >>> p0 = numpy.array([4, 4, 0, 0])
    >>> p1 = numpy.array([1, 1, 2, 4])
    >>> mutual_info(p0, p1)
    0.548794865608

    """

    if weights is None:
        t0 = p0.sum()
        t1 = p1.sum()
    else:
        t0, t1 = weights
    total = t0 + t1

    # entropy of the input variable
    ent = entropy(p0 + p1)
    # entropy of the input variable, conditioned on output value
    cond_ent = (t0*entropy(p0) + t1*entropy(p1))/float(total)
    # mutual information between the pair of variables and the
    # output value
    minfo = (ent - cond_ent)
    # Allows for small numerical errors that might make
    # MI go negative
    if minfo < 0 and abs(minfo/ent) < 1e-6:
        minfo = 0
    if minfo < 0:
        emsg = "Mutual Info. < 0!\nH(X) = %s\nH(X|Y)  = %s\nI(X;Y) = %s\n"
        raise ValueError, emsg % (ent, cond_ent, minfo)
    return minfo


def cond_ent_gain(pair, h0, h1):
    # distribution of row (X) variable
    p0row = h0.sum(1)
    p1row = h1.sum(1)
    # conditional entropy for row variable
    cent_row = conditional_entropy(p0row.reshape(2), p1row.reshape(2))

    # distribution of column (Y) variable
    p0col = h0.sum(0)
    p1col = h1.sum(0)
    cent_col = conditional_entropy(p0col.reshape(2), p1col.reshape(2))

    # distribution of XOR(X, Y) variable
    p0xor = numpy.array((h0[0,0]+h0[1,1], h0[0,1]+h0[1,0]))
    p1xor = numpy.array((h1[0,0]+h1[1,1], h1[0,1]+h1[1,0]))
    cent_xor = conditional_entropy(p0xor.reshape(2), p1xor.reshape(2))

    centropy = array((cent_row, cent_col, cent_xor))
    argmax_cent = centropy.argmax()
    if not argmax_cent == 2:
        if argmax_cent == 1:
            # minimal conditional entropy is achieved by keeping row and
            # replacing column
            repl = pair[1]
        if argmax_cent == 0:
            # minimal conditional entropy is achieved by keeping column and
            # replacing row
            repl = pair[0]
    gain = centropy.max() - centropy.min()
    return gain, repl



def _check_dataset(data, freq0, freq1):
    """Verifies the shape of a data set"""

    if data.ndim != 2:
        raise ValueError, "Input data array of wrong shape %s" % (data.shape,)
    if freq0.shape != freq1.shape:
        raise ValueError, \
            "Shapes of frequency vectors do not match: %s; %s" % \
            (freq0.shape, freq1.shape)
    # npoints := number of different examples
    # veclen := number of elements in each example
    npoints, veclen = data.shape
    freqpoints, = freq0.shape
    if freqpoints != npoints:
        raise ValueError, "Number of data points and number of frequency \
             entries do not match: %d data points; %d frequency entries" % \
             (npoints, freqpoints)
    del freqpoints


def minimize_conditional_info(data, freq0, freq1, max_iter=-1):
    """Performs minimization of mutual information conditioned on the output
    variable.

    :Inputs:
    data:  matrix of binary examples; each example is a row in the matrix
    freq0, freq1:
    max_iter = maximum number of iterations

    """
    _check_dataset(data, freq0, freq1)
    # npoints := number of different examples
    # veclen := number of elements in each example
    npoints, veclen = data.shape

    # initial list of clusters; each element gives rise to
    # a cluster of its own
    clusters = [(el,) for el in range(veclen)]
    if max_iter < 0:
        max_iter = 2*(veclen**2)
    iterations = 0

    # boolean version of data matrix
    booldata = data > 0

    T0 = freq0.sum()
    T1 = freq1.sum()

    # matrix of bidimensional histograms, given output=0
    h2d_0 = hist2dmatrix(data, freq0)
    # matrix of bidimensional histograms, given output=1
    h2d_1 = hist2dmatrix(data, freq1)

    # matrix which holds transformation gains
    gainmatrix = numpy.zeros((veclen-1, veclen), dtype=numpy.float32)
    # matrix whic holds the index to the variables which should
    # be replaced
    replace_matrix = -numpy.ones((veclen-1, veclen), dtype=numpy.int16)

    for pair in _orderedpairs(veclen):
        gainmatrix[pair], replace_matrix[pair] = cond_ent_gain(pair,
                                                              h2d_0[pair],
                                                              h2d_1[pair])
    numrows, numcols = gainmatrix.shape

    # sequential indices
    seq = numpy.arange(numcols)


    while iterations < max_iter:
        # for each column, indexset records the row which achieves the
        # highest gain
        indexset = gainmatrix.argmax(axis=0)
        # transform variable
        maxcol = -1
        try:
            maxcol = gainmatrix[indexset, seq].argmax()
        except ValueError:
            print "indexset.shape = ", indexset.shape
            print "seq.shape = ", seq.shape
        maxrow = indexset[maxcol]
        maxgain = gainmatrix[maxrow, maxcol]
        if maxgain < 0:
            print "Cannot minimize conditional information further; breaking."
            break

        replaced = replace_matrix[maxrow, maxcol]
        # exclusive or operation
        newvalue = booldata[:, maxrow] ^ booldata[:, maxcol]
        booldata[:, replaced] = newvalue

        for lowindex in range(replaced):
            h2d_0[lowindex, replaced] = \
                       _histogram2d(booldata[:, (lowindex, replaced)], freq0)
            h2d_1[lowindex, replaced] = \
                       _histogram2d(booldata[:, (lowindex, replaced)], freq1)
            gainmatrix[lowindex, replaced], replace_matrix[lowindex, replaced] = \
                cond_ent_gain((lowindex, replaced), h2d_0[lowindex, replaced],
                                                    h2d_1[lowindex, replaced])

        for hindex in range(replaced+1, numcols):
            h2d_0[replaced, hindex] = \
                       _histogram2d(booldata[:, (replaced, hindex)], freq0)
            h2d_1[replaced, hindex] = \
                       _histogram2d(booldata[:, (replaced, hindex)], freq1)
            gainmatrix[replaced, hindex], replace_matrix[replaced, hindex] = \
                cond_ent_gain((replaced, hindex), h2d_0[replaced, hindex],
                                                  h2d_1[replaced, hindex])

        # update loop control variables
        iterations += 1
        clusters[int(replaced)] = _cluster_fusion(clusters[maxrow], clusters[maxcol])
    return clusters, booldata.astype(data.dtype), freq0, freq1, iterations

def interactions(data, freq0, freq1):
    if not data.ndim == 2:
        raise ValueError, \
              "Data array of wrong shape: %s. Expected 2 dimensional array." \
              % (data.shape,)
    npoints, veclen = data.shape
    if data.max() != 1 or data.min() != 0:
        raise ValueError, "Expected a data matrix of zeros and ones only."

    T0 = freq0.sum()
    T1 = freq1.sum()
    Total = T0 + T1

    # entropy of output variable
    hy = _bin_entr(float(T1)/Total)
    print 'Output entropy: ', hy

    # number of pairs (output=0, x = 1)
    y0_x1 = numpy.dot(data.transpose(), freq0)
    # number of pairs (output=1, x = 1)
    y1_x1 = numpy.dot(data.transpose(), freq1)

    y0_x0 = T0 - y0_x1
    assert (y0_x0 >= 0).all(), \
           "Count of (y=0 | x=1) pairs is bigger than count of (y=0)"
    y1_x0 = T1 - y1_x1
    assert (y1_x0 >= 0).all(), \
           "Count of (y=1 | x=1) pairs is bigger than count of (y=1)"

    # number of ones per component
    x1 = y0_x1 + y1_x1
    # number of zeros per component
    x0 = y0_x0 + y1_x0

    assert ((x0+x1) == Total).all(), \
           "Count of elements does not match sum of frequencies!"

    idx0nz = x0.nonzero()[0]
    idx1nz = x1.nonzero()[0]
    x0nz = x0[idx0nz]
    x1nz = x1[idx1nz]

    # entropy of y conditioned on x
    hy_x = numpy.zeros(x0.shape, dtype=numpy.float32)
    hy_x[idx0nz] += \
           (x0nz/float(Total))*_vec_bin_entr(y1_x0[idx0nz].astype(float)/x0nz)
    hy_x[idx1nz] += \
           (x1nz/float(Total))*_vec_bin_entr(y1_x1[idx1nz].astype(float)/x1nz)

    hx = _vec_bin_entr(x0.astype(float)/(x0+x1).astype(float))
    # mutual  information between y and x
    mi_xy = hy - hy_x

    infomat = zeros((veclen, veclen), numpy.float32)
    # joint entropy matrix
    jentmat = zeros((veclen, veclen), numpy.float32)

    for x1, x2 in _orderedpairs(veclen):
        nzx1 = data[:, x1] > 0
        nzx2 = data[:, x2] > 0
        i00 = numpy.negative(nzx1 | nzx2)
        i01 = numpy.negative(nzx1) & nzx2
        i10 = nzx1 & numpy.negative(nzx2)
        i11 = nzx1 & nzx2
        h0 = numpy.array((freq0[i00].sum(), freq0[i01].sum(), \
                          freq0[i10].sum(), freq0[i11].sum()))
        h1 = numpy.array((freq1[i00].sum(), freq1[i01].sum(), \
                          freq1[i10].sum(), freq1[i11].sum()))
        infomat[x1, x2] = mutual_info(h0, h1)
        infomat[x2, x1] = infomat[x1, x2]

        jentmat[x1, x2] = entropy(numpy.concatenate((h0, h1)))
        jentmat[x2, x1] = jentmat[x1, x2]

    intermat = infomat.copy()
    idx = numpy.arange(veclen)

    intermat[:, idx] -= mi_xy[idx]
    intermat = intermat.transpose()
    intermat[:, idx] -= mi_xy[idx]
    # intermat = intermat.transpose()
    intermat[idx, idx] = -mi_xy[idx]
    jentmat[idx, idx] = hx + hy_x
    normint = 1 - numpy.abs(intermat)/jentmat

    return intermat, normint, infomat, mi_xy, jentmat


#
# *** Now we begin test definitions ***
#

_data_length_1 = 4

_test_data1 = numpy.array([(0, 1, 0, 1),
               (0, 0, 0, 1),
               (0, 0, 1, 0),
               (0, 1, 0, 0)])
_test_freq1 = numpy.array([1, 2, 3, 4])

_expected1_1d = {(0,0):10,
                 (0,1):0,
                 (1,0):5,
                 (1,1):5,
                 (2,0):7,
                 (2,1):3,
                 (3,0):7,
                 (3,1):3,
                 }
_expected1_1d = _array_from_dict(_expected1_1d, (4,2))
_expected1_2d = {(0,1, 0,0):5,
                 (0,1, 0,1):5,
                 (0,1, 1,0):0,
                 (0,1, 1,1):0,
                 (0,2, 0,0):7,
                 (0,2, 0,1):3,
                 (0,2, 1,0):0,
                 (0,2, 1,1):0,
                 (0,3, 0,0):7,
                 (0,3, 0,1):3,
                 (0,3, 1,0):0,
                 (0,3, 1,1):0,
                 (1,2, 0,0):2,
                 (1,2, 0,1):3,
                 (1,2, 1,0):5,
                 (1,2, 1,1):0,
                 (1,3, 0,0):3,
                 (1,3, 0,1):2,
                 (1,3, 1,0):4,
                 (1,3, 1,1):1,
                 (2,3, 0,0):4,
                 (2,3, 0,1):3,
                 (2,3, 1,0):3,
                 (2,3, 1,1):0
                 }
_expected1_2d = _array_from_dict(_expected1_2d, (3,4,2,2))

_expected1_3d = {(0,1,2, 0,0,0):2,
                 (0,1,2, 0,0,1):3,
                 (0,1,2, 0,1,0):5,
                 (0,1,2, 0,1,1):0,
                 (0,1,2, 1,0,0):0,
                 (0,1,2, 1,0,1):0,
                 (0,1,2, 1,1,0):0,
                 (0,1,2, 1,1,1):0,
                 #
                 (0,1,3, 0,0,0):3,
                 (0,1,3, 0,0,1):2,
                 (0,1,3, 0,1,0):4,
                 (0,1,3, 0,1,1):1,
                 (0,1,3, 1,0,0):0,
                 (0,1,3, 1,0,1):0,
                 (0,1,3, 1,1,0):0,
                 (0,1,3, 1,1,1):0,
                 #
                 (0,2,3, 0,0,0):4,
                 (0,2,3, 0,0,1):3,
                 (0,2,3, 0,1,0):3,
                 (0,2,3, 0,1,1):0,
                 (0,2,3, 1,0,0):0,
                 (0,2,3, 1,0,1):0,
                 (0,2,3, 1,1,0):0,
                 (0,2,3, 1,1,1):0,
                 #
                 (1,2,3, 0,0,0):0,
                 (1,2,3, 0,0,1):2,
                 (1,2,3, 0,1,0):3,
                 (1,2,3, 0,1,1):0,
                 (1,2,3, 1,0,0):4,
                 (1,2,3, 1,0,1):1,
                 (1,2,3, 1,1,0):0,
                 (1,2,3, 1,1,1):0,
                 }
_expected1_3d = _array_from_dict(_expected1_3d, (2,3,4,2,2,2))

_data_length_2 = 3
_test_data2 = numpy.array([(0,0,0),
               (0,0,1),
               (0,1,0),
               (0,1,1),
               (1,0,0),
               (1,0,1),
               (1,1,0),
               (1,1,1)
              ])
_test_freq2 = numpy.ones(8)

_expected2_1d = {(0,0):4,
                 (0,1):4,
                 (1,0):4,
                 (1,1):4,
                 (2,0):4,
                 (2,1):4,
                }
_expected2_1d = _array_from_dict(_expected2_1d, (3,2))

_expected2_2d = {(0,1, 0,0):2,
                 (0,1, 0,1):2,
                 (0,1, 1,0):2,
                 (0,1, 1,1):2,
                 (0,2, 0,0):2,
                 (0,2, 0,1):2,
                 (0,2, 1,0):2,
                 (0,2, 1,1):2,
                 (1,2, 0,0):2,
                 (1,2, 0,1):2,
                 (1,2, 1,0):2,
                 (1,2, 1,1):2
                }
_expected2_2d = _array_from_dict(_expected2_2d, (2,3,2,2))

_expected2_3d = {(0,1,2, 0,0,0):1,
                 (0,1,2, 0,0,1):1,
                 (0,1,2, 0,1,0):1,
                 (0,1,2, 0,1,1):1,
                 (0,1,2, 1,0,0):1,
                 (0,1,2, 1,0,1):1,
                 (0,1,2, 1,1,0):1,
                 (0,1,2, 1,1,1):1,
                 }
_expected2_3d = _array_from_dict(_expected2_3d, (1,2,3,2,2,2))

def test_orderedpairs():
    results = []
    length = 8
    errmsg1 = "Non-increasing pair: (%d, %d)"
    errmsg2 = "Repeated result: (%d, %d)"
    for pair in _orderedpairs(length):
        (a,b) = pair
        assert a < b,  errmsg1 % pair
        assert a >= 0
        assert b < length
        assert pair not in results, errmsg2 % pair
        results.append(pair)

def test_orderedtriples():
    results = []
    length = 8
    errmsg1 = "Non-increasing triple: (%d, %d, %d)"
    errmsg2 = "Repeated result: (%d, %d, %d)"
    for triple in _orderedtriples(length):
        (a,b,c) = triple
        assert a < b,  errmsg1 % triple
        assert b < c,  errmsg1 % triple
        assert a >= 0
        assert c < length
        assert triple not in results, errmsg2 % triple
        results.append(triple)


def testentropy():
    assert entropy([10,10]) - 1 < 10**(-6)
    assert entropy([1,0,0,0,0]) == 0


def testhist2dmatrix():
    emsg = "hist2dmatrix failed. Dataset: %d\nExpected: %s\n \
            Obtained: %s"
    hists = hist2dmatrix(_test_data1, _test_freq1)
    comparison = (hists == _expected1_2d)
    if not numpy.all(comparison):
        raise ValueError, emsg % (_expected1_2d, hists)
    hists = hist2dmatrix(_test_data2, _test_freq2)
    comparison = (hists == _expected2_2d)
    if not numpy.all(comparison):
        raise ValueError, emsg % (_expected2_2d, hists)


def testhist2dto1d():
    testparam = ((_expected1_2d, _expected1_1d, _data_length_1), \
                 (_expected2_2d, _expected2_1d, _data_length_2))
    for (h2d, h1d, dlen) in testparam:
        hists = hist2dto1d(h2d, dlen)
        assert numpy.all(hists == h1d)

def testupdateentry():
    test1 = {(0,1, 0,0):4,
             (0,1, 0,1):0,
             (0,1, 1,0):0,
             (0,1, 1,1):4,
             (0,2, 0,0):2,
             (0,2, 0,1):2,
             (0,2, 1,0):2,
             (0,2, 1,1):2,
             (1,2, 0,0):2,
             (1,2, 0,1):2,
             (1,2, 1,0):2,
             (1,2, 1,1):2,
            }
    test1 = _array_from_dict(test1, (2,3,2,2))
    updateentry(test1, (0,1), 0)
    result1 = {(0,1, 0,0):4,
               (0,1, 0,1):4,
               (0,1, 1,0):0,
               (0,1, 1,1):0,
               (0,2, 0,0):2,
               (0,2, 0,1):2,
               (0,2, 1,0):2,
               (0,2, 1,1):2,
               (1,2, 0,0):2,
               (1,2, 0,1):2,
               (1,2, 1,0):2,
               (1,2, 1,1):2,
            }
    result1 = _array_from_dict(result1, (2,3,2,2))
    assert numpy.all(test1 == result1)

    test2 = {(0,1, 0,0):4,
             (0,1, 0,1):0,
             (0,1, 1,0):0,
             (0,1, 1,1):4,
             (0,2, 0,0):2,
             (0,2, 0,1):2,
             (0,2, 1,0):2,
             (0,2, 1,1):2,
             (1,2, 0,0):2,
             (1,2, 0,1):2,
             (1,2, 1,0):2,
             (1,2, 1,1):2,
            }
    test2 = _array_from_dict(test2, (2,3,2,2))
    updateentry(test2, (0,1), 1)
    result2 = {(0,1, 0,0):4,
               (0,1, 0,1):0,
               (0,1, 1,0):4,
               (0,1, 1,1):0,
               (0,2, 0,0):2,
               (0,2, 0,1):2,
               (0,2, 1,0):2,
               (0,2, 1,1):2,
               (1,2, 0,0):2,
               (1,2, 0,1):2,
               (1,2, 1,0):2,
               (1,2, 1,1):2,
            }
    result2 = _array_from_dict(result2, (2,3,2,2))
    assert numpy.all(test2 == result2)

def test_info_gain():
    h2d = zeros((1,2,2,2))
    expect0 = 0.0817
    h2d[0,1,0,0] = 30
    h2d[0,1,0,1] = 10
    h2d[0,1,1,0] = 30
    h2d[0,1,1,1] = 50
    g0 = _info_gain(h2d, 0, 1)
    assert abs(max(g0) - expect0) < 0.0001
    # now we swap the main diagonal, the result should be the same
    h2d[0,1,0,0], h2d[0,1,1,1] = h2d[0,1,1,1], h2d[0,1,0,0]
    g0 = _info_gain(h2d, 0, 1)
    assert abs(max(g0) - expect0) < 0.0001

    # now we swap the minor diagonal, the result should be the same
    h2d[0,1,1,0], h2d[0,1,0,1] = h2d[0,1,0,1], h2d[0,1,1,0]
    g0 = _info_gain(h2d, 0, 1)
    assert abs(max(g0) - expect0) < 0.0001

    h2d[0,1,0,0] = 50
    h2d[0,1,0,1] = 10
    h2d[0,1,1,0] = 30
    h2d[0,1,1,1] = 30
    g0 = _info_gain(h2d, 0, 1)
    assert abs(max(g0) - 0.0817) < 0.0001

    h2d[0,1,0,0] = 30
    h2d[0,1,0,1] = 10
    h2d[0,1,1,0] = 50
    h2d[0,1,1,1] = 30
    g1 = _info_gain(h2d, 0, 1)
    assert max(g1) < 0

    # now we swap the minor diagonal, the result should be the same
    h2d[0,1,1,0], h2d[0,1,0,1] = h2d[0,1,0,1], h2d[0,1,1,0]
    g1 = _info_gain(h2d, 0, 1)
    assert max(g1) < 0



def testinfo_gain_matrix():
    datalen = 3
    # best transformation is (A,B,C) -> (XOR(A,B), B, C)
    h1 = {(0,1, 0,0):20,
          (0,1, 0,1):20,
          (0,1, 1,0):0,
          (0,1, 1,1):40,
          (0,2, 0,0):20,
          (0,2, 0,1):20,
          (0,2, 1,0):20,
          (0,2, 1,1):20,
          (1,2, 0,0):20,
          (1,2, 0,1):20,
          (1,2, 1,0):20,
          (1,2, 1,1):20,
         }
    h1 = _array_from_dict(h1, (2,3,2,2))
    # best transformation is (A,B,C) -> (A, XOR(A,B), C)
    h2 = {(0,1, 0,0):20,
          (0,1, 0,1):0,
          (0,1, 1,0):20,
          (0,1, 1,1):40,
          (0,2, 0,0):20,
          (0,2, 0,1):20,
          (0,2, 1,0):20,
          (0,2, 1,1):20,
          (1,2, 0,0):20,
          (1,2, 0,1):20,
          (1,2, 1,0):20,
          (1,2, 1,1):20,
         }
    h2 = _array_from_dict(h2, (2,3,2,2))
    # best transformation is (A,B,C) -> (A, XOR(B,C), C)
    h3 = {(0,1, 0,0):20,
          (0,1, 0,1):20,
          (0,1, 1,0):20,
          (0,1, 1,1):20,
          (0,2, 0,0):20,
          (0,2, 0,1):20,
          (0,2, 1,0):20,
          (0,2, 1,1):20,
          (1,2, 0,0):20,
          (1,2, 0,1):0,
          (1,2, 1,0):20,
          (1,2, 1,1):40,
         }
    h3 = _array_from_dict(h3, (2,3,2,2))

    g1 = info_gain_matrix(h1,datalen)
    g2 = info_gain_matrix(h2,datalen)
    g3 = info_gain_matrix(h3,datalen)

def testupdatehists():
    h3d_1 = {(0,1,2, 0,0,0):10,
            (0,1,2, 0,0,1):10,
            (0,1,2, 0,1,0):10,
            (0,1,2, 0,1,1):10,
            (0,1,2, 1,0,0):0,
            (0,1,2, 1,0,1):0,
            (0,1,2, 1,1,0):20,
            (0,1,2, 1,1,1):20,
            }
    h2d_1 = {(0,1, 0,0):20,
          (0,1, 0,1):20,
          (0,1, 1,0):0,
          (0,1, 1,1):40,
          (0,2, 0,0):20,
          (0,2, 0,1):20,
          (0,2, 1,0):20,
          (0,2, 1,1):20,
          (1,2, 0,0):10,
          (1,2, 0,1):10,
          (1,2, 1,0):30,
          (1,2, 1,1):30,
         }
    updateentry(h2d_1, (0,1), 0)
    updatehists(h3d_1, h2d_1, 1, 0, 2)
    exp2d_1 = {(0,1, 0,0):20,
          (0,1, 0,1):40,
          (0,1, 1,0):0,
          (0,1, 1,1):20,
          (0,2, 0,0):30,
          (0,2, 0,1):30,
          (0,2, 1,0):10,
          (0,2, 1,1):10,
          (1,2, 0,0):10,
          (1,2, 0,1):10,
          (1,2, 1,0):30,
          (1,2, 1,1):30,
         }
    exp3d_1 = {(0,1,2, 0,0,0):10,
            (0,1,2, 0,0,1):10,
            (0,1,2, 0,1,0):20,
            (0,1,2, 0,1,1):20,
            (0,1,2, 1,0,0):0,
            (0,1,2, 1,0,1):0,
            (0,1,2, 1,1,0):10,
            (0,1,2, 1,1,1):10,
            }
    assert h3d_1 == exp3d_1
    assert h2d_1 == exp2d_1


def test_cluster_fusion():
    assert (0, 2, 5) == _cluster_fusion((1,2,3,4,6), (0,1,3,4,5,6))


def _test():
    import doctest
    print "Running 'doctest' tests..."
    doctest.testmod()


if __name__ == "__main__":
    import sys
    print "Testing..."
    if len(sys.argv) == 2 and sys.argv[1] == 'test':
        _test()
    test_orderedtriples()
    test_orderedpairs()
    testhist2dmatrix()
    testhist2dto1d()
    testentropy()
    test_info_gain()
    test_cluster_fusion()
    print "Ok!"
