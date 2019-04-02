import numpy as np
from support import pdfMetalog, quantileMetalog

# YEAR: 2019
# COPYRIGHT HOLDER: Travis Jefferies

# Adapted from the R package `rmetalog` by Isaac J. Faber


def pdfMetalogValidation(x):
    """
    pdf validation function - call this feasibility

    :param:     x:      numpy.ndarray:  array of metalog pdf values
    :return:    str:    'yes' if feasible, 'no' if infeasible
    """

    y = np.min(x)
    if y >= 0:
        return 'yes'
    if y < 0:
        return 'no'


def flatten(items, seqtypes=(list, tuple)):
    """
    Helper function used to flatten arrays of arrays

    :param:     items:      list/tuple:  array-like object containing other array-like objects
    :return:    items:      list:       "flattened" 1D version of input
    """

    for i, x in enumerate(items):
        while i < len(items) and isinstance(items[i], seqtypes):
            items[i:i + 1] = items[i]
    return items


def pdf_quantile_builder(temp, y, term_limit, bounds, boundedness):
    """
    Given an a vector, produce a pd.DataFrame with pdf and quantile values for cumulants

    TODO: write docstring
    """

    # TODO: write assertions

    myList = dict()

    # Build pdf
    m = pdfMetalog(temp, y[0], term_limit, bounds=bounds,
                   boundedness=boundedness)
    m = m.tolist()

    for j in range(1, len(y)):
        tempPDF = pdfMetalog(temp, y[j], term_limit, bounds=bounds,
                             boundedness=boundedness)
        m = m + tempPDF.tolist()

    # Build quantile values
    M = quantileMetalog(temp, y[0], term_limit, bounds=bounds,
                        boundedness=boundedness)
    M = M.tolist()

    for j in range(1, len(y)):
        tempQant = quantileMetalog(temp, y[j], term_limit, bounds=bounds,
                                   boundedness=boundedness)

        M = M + tempQant.tolist()

    # Add trailing and leading zero's for pdf bounds

    if boundedness == 'sl':
        m = [0] + m
        M = [bounds[0]] + M

    if boundedness == 'su':
        m = m + [0]
        M = [M] + [bounds[1]]

    if boundedness == 'b':
        m = [0] + [m] + [0]
        M = [bounds[1]] + [M] + [bounds[2]]

    # Add y values for bounded models
    if boundedness == 'sl':
        y = [0] + [y]

    if boundedness == 'su':
        y = [y] + [1]

    if boundedness == 'b':
        y = [0] + [y] + [1]

    # flatten the array - could get around this if I used better data structures
    # data structure design choices are based on an intent to try to mirror the R package as closely as possible
    # m = list(flatten(m))
    # M = list(flatten(M))

    # filter out nans
    m = list(filter(lambda v: v == v, m))
    M = list(filter(lambda v: v == v, M))

    myList['m'] = m
    myList['M'] = M
    myList['y'] = y

    # PDF validation
    myList['valid'] = pdfMetalogValidation(myList['m'])

    return myList

