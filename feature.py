# -*- coding: latin-1 -*-

# Author: Carlos Santos

# WARNING: this will trigger python 3k division (i.e. true division).
# Remember using // for integer division
from __future__ import division

"""Fast selection of binary features (based on work by François Fleuret).

"""

#TODO: Fix doctests, should not rely on printing floating point values

__docformat__ = 'reStructuredText'

__all__ = ["output_mi", "element_entropy", "joint_entropy", "cmim",
           "frequency_count"]

import numpy as np
from numpy import dot
from itanalysis import _histogram2d as hist2d
from itanalysis import mutual_info
from itanalysis import _cluster_fusion as cluster_fusion
import heapq as hq

# float limits
_flimits = np.finfo(np.float)

class Feature(object):
    """Represents candidate features"""
    def __init__(self, position, score, outmi=0, cluster=None):
        self.pos = position
        self.score = score
        self.outmi = outmi
        if not cluster:
            self.cluster = (self.pos,)
        else:
            self.cluster = tuple(sorted(cluster))
        self.min_dep_cluster = self.cluster
        # index of the (already chosen) feature which should be compared
        # next
        self.next_index = 0

    def __cmp__(self, other):
        return cmp(self.score, other.score)

    def update_cluster(self, other):
        self.cluster = cluster_fusion(self.cluster, other.cluster)

    def __str__(self):
        return str(self.cluster)

    def __repr__(self):
        params = (repr(self.position), repr(self.score), repr(self.cluster))
        return "Feature(position= %s, score=%s, cluster= %s)" % params

def entropy_log2(x):
    """Modified log function for entropy calculation. Returns normal log2 if
    the argument is strictly positive or zero if the argument is zero.
    Negative values will raise a warning."""
    if (x<0).any():
        print "Negative value in distribution: "
        print "Array:"
        print x
        print x[x<0]
        raise ValueError("Counting data cannot be negative")
    if x.all():
        return np.log2(x)
    else:
        result = np.zeros(x.shape)
        nz = x.nonzero()
        # print "nz: ", nz
        # print "data: ", x
        result[nz] = np.log2(x[nz])
        return result

def element_entropy(data, count):
    """Entropy of a given element.

    >>> element_entropy([0, 1], [10, 10])
    1.0
    >>> element_entropy([0, 1], [0, 10])
    0.0


    """
    Total = np.sum(count)
    T1 = dot(np.transpose(data), count)
    T0 = Total - T1
    # empirical distribution
    p = np.array((T0, T1))
    entropy = np.log2(Total) - ((p/float(Total))*entropy_log2(p)).sum()
    return entropy

def joint_entropy(candidate, selected, freq0, freq1):
    """

    >>> import numpy as np
    >>> x1 = np.array([0, 0, 1, 1])
    >>> x2 = np.array([0, 1, 0, 1])
    >>> freq0 = np.array([0, 1, 1, 0])
    >>> freq1 = np.array([1, 0, 0, 1])
    >>> joint_entropy(x1, x2, freq0, freq1)
    (2.0, 2.0, 2.0, 1.0)

    """
    T0 = np.sum(freq0)
    T1 = np.sum(freq1)
    Total = T0 + T1
    logTotal = np.log2(Total)
    index = 2*candidate + selected
    hist_y0 = np.zeros(4, dtype=np.int32)
    hist_y1 = np.zeros(4, dtype=np.int32)
    for i in range(4):
        hist_y0[i] = freq0[index==i].sum()
        hist_y1[i] = freq1[index==i].sum()
    # three-way joint entropy
    joint_entropy3 = logTotal - (hist_y0*entropy_log2(hist_y0)/Total).sum() \
                              - (hist_y1*entropy_log2(hist_y1)/Total).sum()
    hist2 = hist_y0 + hist_y1
    joint_entropy2 = logTotal - (hist2*entropy_log2(hist2)/Total).sum()
    hist_sel_y0 = np.array((hist_y0[0] + hist_y0[2], hist_y0[1] + hist_y0[3]))
    hist_sel_y1 = np.array((hist_y1[0] + hist_y1[2], hist_y1[1] + hist_y1[3]))
    ent_y_sel = logTotal \
                - (hist_sel_y0*entropy_log2(hist_sel_y0)/Total).sum() \
                - (hist_sel_y1*entropy_log2(hist_sel_y1)/Total).sum()
    hist_sel = hist_sel_y0 + hist_sel_y1
    ent_sel = logTotal - (hist_sel*entropy_log2(hist_sel)/Total).sum()
    return joint_entropy3, joint_entropy2, ent_y_sel, ent_sel

def joint_ent_element_output(data, freq0, freq1):
    T0 = freq0.sum()
    T1 = freq1.sum()
    Total = T0 + T1
    # empirical distribution of output variable
    pout = np.array([T0, T1])

    # frequency of output variable conditioned data
    f0_1 = dot(data.transpose(), freq0)
    f0_0 = T0 - f0_1
    f1_1 = dot(data.transpose(), freq1)
    f1_0 = T1 - f1_1
    logT0 = np.log2(T0)
    logT1 = np.log2(T1)
    logTotal = np.log2(Total)
    # empirical joint distribution
    pjoint = np.array((f0_0, f1_0, f0_1, f1_1))
    joint_entropy = logTotal - \
                    np.sum((pjoint/float(Total))*entropy_log2(pjoint))
    return joint_entropy

def output_mi(data, freq0, freq1):
    """Calculates the mutual information between each component of the data
    and the output variable.

    :Input:

    data
    freq0
    freq1


    >>> import numpy as np
    >>> data = np.array([ [0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 1, 1] ], dtype = np.int8)
    >>> f0 = np.array([0, 1, 0, 1], dtype=np.int8)
    >>> f1 = 1 - f0
    >>> mi, out_ent, elem_ent, joint_ent = output_mi(data, f0, f1)
    >>> mi
    array([ 0.3112781,  0.       ,  1.       ])


    """
    T0 = freq0.sum()
    T1 = freq1.sum()
    Total = T0 + T1
    #print T0, T1, Total

    # empirical distribution of output variable
    pout = np.array([T0, T1])

    # frequency of output variable conditioned on each element
    f0_1 = dot(data.transpose(), freq0)
    f0_0 = T0 - f0_1
    f1_1 = dot(data.transpose(), freq1)
    f1_0 = T1 - f1_1

    #print f0_0, ";\n", f0_1, ";\n", f1_0, ";\n", f1_1

    logT0 = np.log2(T0)
    logT1 = np.log2(T1)
    logTotal = np.log2(Total)

    output_entropy = logTotal - ((pout/float(Total))*entropy_log2(pout)).sum()
    # empirical distribution of elements
    pelem = np.vstack((f0_0 + f1_0, f0_1 + f1_1))
    # empirical joint distribution
    pjoint = np.vstack((f0_0, f1_0, f0_1, f1_1))

    elem_entropy = logTotal - \
                   np.sum((pelem/float(Total))*entropy_log2(pelem), axis=0)

    joint_entropy = logTotal - \
                    np.sum((pjoint/float(Total))*entropy_log2(pjoint), axis=0)

    mutual_info = output_entropy + elem_entropy - joint_entropy
    return mutual_info, output_entropy, elem_entropy, joint_entropy

def cmim_score_func(candidate, selected, freq0, freq1):
    hjoint3, hsel_cand, hsel_y, hsel = joint_entropy(candidate, selected, freq0, freq1)
    score = hsel_cand - hjoint3 - hsel + hsel_y
    if score < -100*_flimits.eps:
        print "Error: Negative score"
        print "Score: ", score
        print "freq0: max= %f, min = %f" % (freq0.max(), freq0.min())
        print "freq1: max= %f, min = %f" % (freq1.max(), freq1.min())
        print "hsel_cand: ", hsel_cand
        print "hjoint3: ", hjoint3
        print "hsel: ", hsel
        print "hsel_y: ", hsel_y
        raise ValueError("Negative score.")
    return score

def limited_score_func(candidate, selected, freq0, freq1):
    """Upper limited score function; does not take into account sinergy
    between features"""
    hjoint3, hsel_cand, hsel_y, hsel = joint_entropy(candidate, selected, freq0, freq1)
    score_sinergy = hsel_cand - hjoint3 - hsel + hsel_y
    score = min(score_sinergy, candidate.outmi)
    assert score > 0
    return score

def _safe_min(features, maxelements):
    if features is None:
        return maxelements
    else:
        return min(maxelements, features)

def default_add(featurelist, newfeature, dataset):
    featurelist.append(newfeature)

def _max_info_add_feature(featurelist, newfeature, dataset):
    data, freq0, freq1 = dataset
    for selected in featurelist:
        fidx, selidx = newfeature.pos, selected.pos
        iv = _calc_interactions(newfeature, selected, data, freq0, freq1)
        if iv.mi_xor >  newfeature.outmi:
            # Exclusive OR of data: modulo-2 addition
            data[:, fidx] = \
                np.mod(data[:, fidx] + data[:, selidx], 2).astype(data.dtype)
            # update mutual information
            newfeature.outmi = iv.mi_xor
            # update clusters
            newfeature.update_cluster(selected)
    featurelist.append(newfeature)

def make_fs(forward, add_feature=default_add):
    """Creata a cmim-like feature selection function."""
    def internal_cmim(data, freq0, freq1, numfeatures=None):
        # out_mi:   array; mutual information between output variable and
        #           each element
        # out_ent:  scalar; entropy of output variable
        # elem_ent: array: entropy of each element
        # _ : array; joint entropy between output and each element
        out_mi, out_ent, elem_ent, _ = output_mi(data, freq0, freq1)
        num_elements = len(elem_ent)
        num_features = _safe_min(numfeatures, num_elements)
        #print num_elements, num_features, numfeatures
        # the first feature to be included is the most informative about
        # the output
        features = [Feature(out_mi.argmax(), score=out_mi.max(), outmi=out_mi.max())]
        first_idx = out_mi.argmax()
        candidate_indices = range(first_idx) + range(1+first_idx, num_elements)
        candidates = [Feature(pos, score=min(out_ent, elem_ent[pos]),
                      outmi=out_mi[pos]) for pos in candidate_indices]
        del candidate_indices
        dataset = (data, freq0, freq1)
        for num_selected in range(1, num_features):
            curr_best = candidates[0]
            forward(curr_best, features, dataset, 0)
            for n in range(1, len(candidates)):
                next_cand = candidates[n]
                forward(next_cand, features, dataset, curr_best.score)
                if next_cand.score > curr_best.score:
                    curr_best = next_cand
            candidates.remove(curr_best)
            #print "Appending features"
            add_feature(features, curr_best, dataset)
        indices = [el.pos for el in features]
        reordered_data = data[:, indices]
        return indices, features, reordered_data
    return internal_cmim

def _candidate_forward_cmim(candidate, features, dataset, score):
    data, freq0, freq1 = dataset
    num_selected = len(features)
    while candidate.score > score and candidate.next_index < num_selected:
        selected = features[candidate.next_index]
        canddata, seldata = data[:, candidate.pos], data[:, selected.pos]
        newscore = cmim_score_func(canddata, seldata, freq0, freq1)
        candidate.score = min(candidate.score, newscore)
        candidate.next_index += 1

class InteractionValues(object):
    def __init__(self, mi_pair, mi_xor):
        self.mi_pair = mi_pair
        self.mi_xor = mi_xor

def _calc_interactions(candidate, selected, data, freq0, freq1):
    pair = (candidate.pos, selected.pos)
    h0 = hist2d(data[:,pair], freq0)
    h1 = hist2d(data[:,pair], freq1)
    mi_y_pair = mutual_info(h0.reshape(4), h1.reshape(4))
    # distribution of XOR(n, next) variable
    p0xor = np.array((h0[0,0]+h0[1,1], h0[0,1]+h0[1,0]))
    p1xor = np.array((h1[0,0]+h1[1,1], h1[0,1]+h1[1,0]))
    mi_y_xor = mutual_info(p0xor.reshape(2), p1xor.reshape(2))
    return InteractionValues(mi_y_pair, mi_y_xor)

def _candidate_forward_min_info(candidate, features, dataset, score):
    data, freq0, freq1 = dataset
    num_selected = len(features)
    while candidate.score > score and candidate.next_index < num_selected:
        selected = features[candidate.next_index]
        candidx, selidx = candidate.pos, selected.pos
        iv = _calc_interactions(candidate, selected, data, freq0, freq1)
        if iv.mi_xor <  candidate.outmi:
            # Exclusive OR of data: modulo-2 addition
            data[:, candidx] = \
                np.mod(data[:, candidx] + data[:, selidx], 2).astype(data.dtype)
            # update mutual information
            candidate.outmi = iv.mi_xor
            # update clusters
            candidate.update_cluster(selected)
        newscore = iv.mi_pair - selected.outmi
        candidate.score = min(candidate.score, newscore)
        candidate.next_index += 1



def _heap_cmim(data, freq0, freq1, numfeatures=None):
    # out_mi:   array; mutual information between output variable and
    #           each element
    # out_ent:  scalar; entropy of output variable
    # elem_ent: array: entropy of eache element
    # j_ent: array; joint entropy between output and each element
    out_mi, out_ent, elem_ent, j_ent = output_mi(data, freq0, freq1)
    num_elements = len(elem_ent)
    num_features = _safe_min(numfeatures, num_elements)
    # the first feature to be included is the most informative about
    # the output
    features = [Feature(position=out_mi.argmax(), score=out_mi.max(),
                        outmi=out_mi.max())]
    indices = [out_mi.argmax()]
    candidate_indices = range(indices[0]) + range(1+indices[0], num_elements)
    candidates = [Feature(position=i, score=min(out_ent, elem_ent[i]),
                  outmi=out_mi[i])  for i in candidate_indices]
    del candidate_indices
    hq.heapify(candidates)
    dataset = (data, freq0, freq1)
    for num_selected in range(1, num_features):
        best_candidate = hq.heappop(candidates)
        _candidate_forward_cmim(best_candidate, features, dataset, 0)
        next_candidate = hq.heappop(candidates)
        while best_candidate.score < next_candidate.score:
            _candidate_forward_cmim(next_candidate, features, dataset,
                                 best_candidate.score)
            if best_candidate.score < next_candidate.score:
                best_candidate, next_candidate = best_candidate, next_candidate
            hq.heappush(candidates, next_candidate)
            next_candidate = hq.heappop(candidates)
        indices.append(best_candidate.pos)
        features.append(best_candidate)
    reordered_data = data[:, indices]
    return indices, features, reordered_data

def cmim(data, freq0, freq1, numfeatures=None):
    """Conditional mutual information feature selection.

    Parameters
    -----------------------------

    data: array like of shape [n, m]
        Input patterns, each row is a different binary pattern.
    freq0: array like of n elements
        Frequency associated with the occurrence of label 0
    freq1: array like of n elements
        Frequency associated with the occurrence of label 1
    numfeatures: int
        Maximum number of features returned. If it set to None, all features will be returned, in decreasing order
        of CMIM score.

    Returns
    -------------------------------------------

    idx: sequence of integers.
       The sequence of indices indicating the selected features, in decreasing order of CMIM score.
    features: sequence of 'Feature' objects.
       Sequence of features, in the same order as 'idx'. This data is useful in order to recover the score
       of each feature.
    new_data: array like of shape [n, m]
       The same input patterns contained in 'data', but with the columns reordered so the best feature is found in
       column zero, second best feature is found in column one, and so on.

    """
    cmim_function = make_fs(_candidate_forward_cmim)
    idx, features, new_data = cmim_function(data, freq0, freq1, numfeatures)
    return idx, features, new_data


def _integer_histogram(data):
    """Calculates histogram for integer data, assuring data each histogram
    bin corresponds to one integer value.

    Parameters
    ------------------------------

    data : array
        Integer data for histogram computation

    Returns
    ------------------------------

    (count, bin_centers)

    count: array
        Count of data
    bin_centers : array
        Centers of bins

    Examples
    ------------------------------

    >>> import numpy
    >>> data = numpy.array([1, 1, 3, 5, 5, 5])
    >>> h, x = _integer_histogram(data)
    >>> x
    array([1, 3, 5])
    >>> h
    array([2, 1, 3])

    """
    min_val, max_val = data.min(), data.max()
    bin_centers = np.unique(data).astype(int)
    hist_bins = bin_centers - 0.5
    hist_range = min_val - 0.5, max_val + 0.5
    count, left_edges = np.histogram(data, bins=hist_bins, range=hist_range)
    return count, bin_centers


def _as_binary_data(decimal_data, num_bits, dtype=int):
    """Transforms decimal data into the equivalent binary representation.

    Parameters
    ---------------------

    decimal_data : array
        Decimal array, must be sorted
    num_bits : int
        Number of bits to use in binary representation

    Returns
    --------------------

    binary_data : array
        The data, as a matrix of binary elements.

    Examples
    -------------------

    >>> import numpy
    >>> data = numpy.array([1, 7, 8, 12])
    >>> _as_binary_data(data, 4)
    array([[0, 0, 0, 1],
           [0, 1, 1, 1],
           [1, 0, 0, 0],
           [1, 1, 0, 0]])
    """
    data_len = len(decimal_data)
    binary_data = np.zeros((data_len, num_bits), dtype)
    for i in xrange(num_bits):
        row_index = num_bits - 1 - i
        fill_indices = (decimal_data % 2) == 1
        binary_data[fill_indices, row_index] = 1
        # integer division
        decimal_data = decimal_data//2
    return binary_data

def _unify_counts(count_zero, x_zero, count_one, x_one):
    """Puts both counts in a common reference.

    Examples
    ---------

    >>> import numpy
    >>> x_zero = numpy.array([2, 3], dtype=numpy.int32)
    >>> count_zero = numpy.array([10, 20], dtype=numpy.int32)
    >>> x_one = numpy.array([1, 3], dtype=numpy.int32)
    >>> count_one = numpy.array([7, 15], dtype=numpy.int32)
    >>> f0, f1, x = _unify_counts(count_zero, x_zero, count_one, x_one)
    >>> x
    array([1, 2, 3])
    >>> f0
    array([ 0, 10, 20])
    >>> f1
    array([ 7,  0, 15])


    """
    x = np.unique(np.concatenate((x_zero, x_one)))
    freq0 = np.zeros_like(x)
    freq1 = np.zeros_like(x)
    indices_zero = np.searchsorted(x, x_zero)
    indices_one = np.searchsorted(x, x_one)
    freq0[indices_zero] = count_zero
    freq1[indices_one] = count_one
    return freq0, freq1, x


def frequency_count(binary_data, binary_class):
    """Calculates frequency counts for a given dataset.

    Parameters:
    ------------------------------------------

    binary_data : array
        The binary data; each row corresponds to one example/instace.
    binary_class : array
        Binary class variable; each entry corresponds to the class assignment
        for one example/instace in ``binary_data''.

    Returns:
    ------------------------------------------

    (data, freq0, freq1)

    data : array
         Shortened version of ``binary_data'', each row corresponds to one
         possible value for the examples.
    freq0 : array
         For each example value in ``data'', it corresponds to the count of the
         occurences of zero-valued class variable
    freq1 : array
         For each example value in ``data'', it corresponds to the count of the
         occurences of one-valued class variable


    Examples
    ---------------------------------------------

    >>> import numpy
    >>> binary_data = numpy.array([(1, 0, 0, 0), (0, 0, 1, 0), (0, 1, 0, 1)])
    >>> binary_class = numpy.array([0, 1, 0])
    >>> d, freq0, freq1 = frequency_count(binary_data, binary_class)
    >>> d
    array([[0, 0, 1, 0],
           [0, 1, 0, 1],
           [1, 0, 0, 0]])
    >>> freq0
    array([0, 1, 1])
    >>> freq1
    array([1, 0, 0])

    """
    num_examples, num_features = binary_data.shape
    pow_2 = 2**np.arange(num_features-1, -1, -1)
    decimal_data = np.dot(binary_data, pow_2)
    decimal_data_zero = decimal_data[binary_class == 0]
    decimal_data_one = decimal_data[binary_class == 1]
    count_zero, x_zero = _integer_histogram(decimal_data_zero)
    count_one, x_one = _integer_histogram(decimal_data_one)
    freq0, freq1, x = _unify_counts(count_zero, x_zero, count_one, x_one)
    data = _as_binary_data(x, num_features)
    return data, freq0, freq1


def test_frequency_count():
    binary_data = np.array([(0, 0, 1, 0, 1), #0
                            (1, 1, 1, 0, 1), #0
                            (0, 0, 1, 0, 1), #1
                            (1, 1, 1, 0, 1), #1
                            (0, 0, 1, 0, 1), #1
                            (0, 1, 1, 0, 1), #0
                            (0, 0, 0, 0, 1), #0
                            (0, 1, 1, 0, 1), #0
                            (0, 0, 0, 0, 1), #1
                            (0, 0, 1, 0, 1), #0
                            (0, 0, 1, 0, 1), #0
                            (1, 1, 1, 1, 1), #0
                            (1, 1, 1, 1, 1), #1
                            (0, 0, 0, 0, 0), #1
                            (0, 0, 0, 0, 0), #0
                            (0, 0, 0, 0, 0), #1
                            (0, 0, 0, 0, 0), #1
                            (0, 0, 0, 1, 0), #0
                           ], dtype=np.int32)
    binary_class = np.array([0, 0, 1, 1, 1, 0, 0, 0, 1,
                             0, 0, 0, 1, 1, 0, 1, 1, 0])
    expected_x = np.array([(0, 0, 0, 0, 0),
                           (0, 0, 0, 0, 1),
                           (0, 0, 0, 1, 0),
                           (0, 0, 1, 0, 1),
                           (0, 1, 1, 0, 1),
                           (1, 1, 1, 0, 1),
                           (1, 1, 1, 1, 1),
                           ], dtype=np.int32)
    expected_f0 = np.array([1, 1, 1, 3, 2, 1, 1])
    expected_f1 = np.array([3, 1, 0, 2, 0, 1, 1])
    x, f0, f1 = frequency_count(binary_data, binary_class)
    def test_equal(expected, obtained, label, msg):
        success = True
        if not np.all(obtained == expected):
            msg += "Expected %s:\n" % (label,)
            msg += str(expected_x)
            msg += "\nObtained %s:\n" % (label,)
            msg += str(x)
            succes = False
        return success, msg
    flagx, error_msg = test_equal(expected_x, x, "x", "")
    flag0, error_msg = test_equal(expected_f0, f0, "freq0", error_msg)
    flag1, error_msg = test_equal(expected_f0, f0, "freq1", error_msg)
    if not (flagx and flag0 and flag1):
        raise ValueError, error_msg


def _test():
    import doctest
    print "Running 'doctest' tests..."
    doctest.testmod()
    test_frequency_count()


if __name__ == "__main__":
    import sys
    print "Testing..."
    if len(sys.argv) == 2 and sys.argv[1] == 'test':
        _test()



