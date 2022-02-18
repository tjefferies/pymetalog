import numpy as np
from .support import pdfMetalog, quantileMetalog


def pdf_quantile_builder(temp, y, term_limit, bounds, boundedness):
    """Builds the metalog pdf and quantile arrays based on the a coefficients found by fitting metalog distribution.

    Args:
        temp (:obj: `numpy.ndarray` of type float): Array of a coefficients found by fitting metalog distribution.
            - Fit method is specified by metalog.fit_method attribute

        y (:obj: `numpy.ndarray` of type float): Array of bin widths specified for `a` parameter

        term_limit (:obj: `int`): The upper limit of the range of metalog terms to use to fit the data.
            - metalog.term_limit attribute
            - in range [3,30]

        bounds (:obj:`list`): Upper and lower limits to filter the data with before calculating metalog quantiles/pdfs.
            - metalog.bounds attribute
            - Default: [0,1]

        boundedness (:obj: `str`): String that is used to specify the type of metalog to fit.
            - metalog.boundedness attribute

    Returns:
        q_dict (:obj:`dict` with keys ['m', 'M', 'y', 'valid']): Initialized output_dict variable from metalog class.
            - q_dict['m']: (:obj:`numpy.ndarray` of type float): Array of metalog pdf values.
                * Returned by `pdfMetalog` method
                * Influenced by `boundedness` parameter
                * A valid metalog fit will return an array having all elements strictly > 0

            - q_dict['M']: (:obj:`numpy.ndarray` of type float): Array of metalog quantile values.
                * Returned by `quantileMetalog` method
                * Influenced by `boundedness` parameter
                    - `boundedness` = 'sl': Inserts `bounds`[0] to the front of the quantile array
                    - `boundedness` = 'su': Appends `bounds`[1] to the end of the quantile array
                    - `boundedness` = 'b': Inserts `bounds`[0] to the front of the quantile array
                                            and appends `bounds`[1] to the end of the quantile array

            - q_dict['y']: (:obj:`numpy.ndarray` of type float): Array of bin widths specified for the pdfs/quantiles.
                * Influenced by `boundedness` parameter
                    - `boundedness` = 'sl': Inserts `bounds`[0] at the front of the quantile array
                    - `boundedness` = 'su': Appends `bounds`[1] to the end of the quantile array
                    - `boundedness` = 'b': Inserts `bounds`[0] at the front of the quantile array
                                            and appends `bounds`[1] to the end of the quantile array

            - q_dict['valid']: (:obj:`str`): A string indicating if the metalog pdf generated by `pdfMetalog` method is valid or not.
                * If all values in the metalog pdf are >= 0, q_dict['valid'] = 'yes'
                * If any values in the metalog pdf are < 0, q_dict['valid'] = 'no'

    """
    q_dict = {}

    # build pdf
    m = pdfMetalog(temp, y[0], term_limit, bounds=bounds, boundedness=boundedness)

    for j in range(2, len(y) + 1):
        tempPDF = pdfMetalog(
            temp, y[j - 1], term_limit, bounds=bounds, boundedness=boundedness
        )
        m = np.append(m, tempPDF)

    # Build quantile values
    M = quantileMetalog(temp, y[1], term_limit, bounds=bounds, boundedness=boundedness)

    for j in range(2, len(y) + 1):
        tempQant = quantileMetalog(
            temp, y[j - 1], term_limit, bounds=bounds, boundedness=boundedness
        )
        M = np.append(M, tempQant)

    # Add trailing and leading zero's for pdf bounds
    if boundedness == "sl":
        m = np.append(0, m)
        M = np.append(bounds[0], M)

    if boundedness == "su":
        m = np.append(m, 0)
        M = np.append(M, bounds[1])

    if boundedness == "b":
        m = np.append(0, m)
        m = np.append(m, 0)
        M = np.append(bounds[0], M)
        M = np.append(M, bounds[1])

    # Add y values for bounded models
    if boundedness == "sl":
        y = np.append(0, y)

    if boundedness == "su":
        y = np.append(y, 1)

    if boundedness == "b":
        y = np.append(0, y)
        y = np.append(y, 1)

    q_dict["m"] = m
    q_dict["M"] = M
    q_dict["y"] = y

    # PDF validation
    q_dict["valid"] = pdfMetalogValidation(q_dict["m"])

    return q_dict


def pdfMetalogValidation(x):
    """Validation that all calculated metalog pdf values are greater than or equal to 0.

    Args:
        x (:obj: `numpy.ndarray` of type float): Array of metalog pdf values.
            - Returned by `pdfMetalog` method
            - Influenced by `boundedness` parameter

    Returns:
        'yes' | 'no' (:obj:`str`): 'yes' if all elements strictly >= 0, else 'no'.
    """
    y = np.min(x)
    if y >= 0:
        return "yes"
    else:
        return "no"
