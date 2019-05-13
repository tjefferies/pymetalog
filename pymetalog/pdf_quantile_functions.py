import numpy as np
from .support import pdfMetalog, quantileMetalog

def pdf_quantile_builder(temp, y, term_limit, bounds, boundedness):
    """TODO: write docstring

    """
    q_dict = {}

    # build pdf
    m = pdfMetalog(temp, y[0], term_limit, bounds = bounds, boundedness = boundedness)

    for j in range(2,len(y)+1):
        tempPDF = pdfMetalog(temp, y[j-1], term_limit, bounds = bounds, boundedness = boundedness)
        m = np.append(m, tempPDF)

    # Build quantile values
    M = quantileMetalog(temp, y[1], term_limit, bounds=bounds, boundedness=boundedness)

    for j in range(2, len(y) + 1):
        tempQant = quantileMetalog(temp, y[j-1], term_limit, bounds = bounds, boundedness = boundedness)
        M = np.append(M, tempQant)

    # Add trailing and leading zero's for pdf bounds
    if boundedness == 'sl':
        m = np.append(0, m)
        M = np.append(bounds[0], M)

    if boundedness == 'su':
        m = np.append(m, 0)
        M = np.append(M, bounds[1])

    if boundedness == 'b':
        m = np.append(0, m)
        m = np.append(m, 0)
        M = np.append(bounds[0], M)
        M = np.append(M, bounds[1])

    # Add y values for bounded models
    if boundedness == 'sl':
        y = np.append(0, y)

    if boundedness == 'su':
        y = np.append(y, 1)

    if boundedness == 'b':
        y = np.append(0, y)
        y = np.append(y, 1)

    q_dict['m'] = m
    q_dict['M'] = M
    q_dict['y'] = y

    # PDF validation
    q_dict['valid'] = pdfMetalogValidation(q_dict['m'])
    return q_dict

def pdfMetalogValidation(x):
    """TODO: write docstring

    """
    y = np.min(x)
    if (y >= 0):
        return('yes')
    else:
        return('no')
