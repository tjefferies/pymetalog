import numpy as np
import pandas as pd

# YEAR: 2019
# COPYRIGHT HOLDER: Travis Jefferies

# Adapted from the R package `rmetalog` by Isaac J. Faber


def MLprobs(x, step_len):
    """
    TODO: write docstring
    """

    assert isinstance(x, np.ndarray), 'x must be a numpy.array!'
    assert isinstance(step_len, float), 'step_len must be a float (0, 1)!'
    assert 0 < step_len < 1, 'step_len must be a float (0, 1)!'

    l = len(x)
    x = pd.DataFrame(x)
    x.sort_values(inplace=True)
    x['probs'] = 0
    for i in range(l):
        if i == 0:
            x['probs'].values[i] = 0.5 / l
        else:
            x['probs'].values[i] = (x['probs'].values[i - 1] + (1 / l))

    # If the data is very long we down convert to a smaller but representative
    # vector using the step_len default is 0.01 which is a 109 element vector with
    # fine values in the tail (tailstep)

    if len(x) > 100:
        y = np.arange(step_len, 1, step_len)
        tailstep = step_len / 10
        y = [np.arange(tailstep, min(y) - tailstep, tailstep),
             y,
             np.arange(max(y) + tailstep, max(y) + tailstep * 9, tailstep)
             ]
        x_new = np.quantile(x.iloc[:, 0], q=np.array(y))
        x = pd.DataFrame(x_new)
        x['probs'] = np.array(y)

    return x

def diffMatMetalog(term_limit, step_len):
    """
    Function for returning the matrix of differentiation terms

    :param:     term_limit: int:    The number of terms to differentiate
    :param:     step_len:   float:  Training size step
    :return:    new_Diff:   pd.DataFrame:   pd.DataFrame of differentiated terms

    """

    assert isinstance(term_limit, int), 'term_limit must be an int [1, 10)!'
    assert 1 <= term_limit < 10, 'term_limit must be an int [1, 10)!'
    assert isinstance(step_len, float), 'step_len must be a float (0, 1)!'
    assert 0 < step_len < 1, 'step_len must be a float (0, 1)!'

    y = np.arange(step_len, 1, step_len)
    Diff = list()

    for i in y:
        d = i * (1 - i)
        f = i - 0.5
        l = round(np.log(i / (1 - i)), 5) # rounding result to mirror default R output

        # Initiate pdf
        diffVector = 0.0

        # For the first three terms
        x = 1.0 / d
        diffVector = [diffVector, x]

        if term_limit > 2:
            diffVector = diffVector + [(f / d) + l]

        if term_limit > 3:
            diffVector = diffVector + [1]

        # Initalize some counting variables
        e = 1
        o = 1

        # For all other terms greater than 4
        if term_limit > 4:
            for i in range(5, term_limit):
                if i % 2 != 0:
                    # iff odd
                    diffVector = diffVector + [(o + 1) * f ^ o]
                    o = o + 1

                if i % 2 == 0:
                    # iff even
                    diffVector = diffVector + [((f ^ (e + 1)) / d) + (e + 1) * (f ^ e) * l]
                    e = e + 1

        Diff = Diff + [diffVector]

    Diff = np.array(Diff)
    Diff_neg = -Diff
    new_Diff = pd.concat([pd.Series(Diff[:, 0]), pd.Series(Diff_neg[:, 0])], axis=1)

    for c in range(1, len(Diff[0, :])):
        new_Diff = pd.concat([new_Diff, pd.Series(Diff[:, c])], axis=1)
        new_Diff = pd.concat([new_Diff, pd.Series(Diff_neg[:, c])], axis=1)

    return new_Diff

def quantileMetalog(a, y, t, bounds=(), boundedness='u'):
    """
    TODO: write docstring


    :param: a:              numpy.array:    array of a coefficients
    :param: y:              numpy.array:    array of data to fit a metalog to
    :param: t:              int:            number of terms to use in metalog quantile calculations
    :param: bounds:         tuple:          lower/upper bounds placed on the metalog
    :param: boundedness:    str:            'u' for unbounded metalog,
                                            'sl' for strictly lower bounded metalog,
                                            'su' for strictly upper bounded metalog,
                                            'b' for lower/upper bounded metalog
    :return: x:             numpy.ndarray:  array of metalog quantile values

    """

    assert isinstance(a, np.ndarray), 'a must be a numpy.array!'
    assert isinstance(y, np.ndarray), 'y must be a numpy.array!'
    assert isinstance(t, int), 't must be an integer!'
    assert t > 2, 't must be at least 3 terms!'
    assert len(a) == t, 'a must be initialized to have the same number of coefficients as arg t ({})'.format(t)
    assert boundedness in ['u', 'sl', 'su', 'b'], 'boundedness must be in [u, sl, su, b]'

    if boundedness in ['sl', 'su', 'b']:
        assert len(bounds) == 2, "when boundedness in ['sl', 'su', 'b'], you must include feasible bounds!"
        assert bounds[1] > bounds[0], "when boundedness in ['sl', 'su', 'b'], you must include feasible bounds!"

    f = y - 0.5
    l = np.log(y / (1 - y))

    # For the first three terms
    x = a[0] + a[1] * l + a[2] * f * l

    # For the fourth term
    if t > 2:
        x = x + a[3] * f

    # Some tracking variables
    o = 2
    e = 2

    # For all other terms greater than 4
    if t > 3:
        for i in range(4, t):
            if i % 2 == 0:
                x = x + a[i] * np.power(f, e) * l
                e = e + 1

            if i % 2 != 0:
                x = x + a[i] * np.power(f, o)
                o = o + 1

    if boundedness == 'sl':
        x = bounds[0] + np.exp(x)

    if boundedness == 'su':
        x = bounds[1] - np.exp(-x)

    if boundedness == 'b':
        x = (bounds[0] + bounds[1] * np.exp(x)) / (1 + np.exp(x))

    return x


def pdfMetalog(a, y, t, bounds=(), boundedness='u'):
    """
    TODO: write docstring

    :param: a:              numpy.array:    array of a coefficients
    :param: y:              numpy.array:    array of data to fit a metalog to
    :param: t:              int:            number of terms to use in metalog quantile calculations
    :param: bounds:         tuple:          lower/upper bounds placed on the metalog
    :param: boundedness:    str:            'u' for unbounded metalog,
                                            'sl' for strictly lower bounded metalog,
                                            'su' for strictly upper bounded metalog,
                                            'b' for lower/upper bounded metalog
    :return: x:             numpy.ndarray:  array of metalog pdf values
    """

    assert isinstance(a, np.ndarray), 'a must be a numpy.array!'
    assert isinstance(y, np.ndarray), 'y must be a numpy.array!'
    assert isinstance(t, int), 't must be an integer!'
    assert t > 2, 't must be at least 3 terms!'
    assert len(a) == t, 'a must be initialized to have the same number of coefficients as arg t ({})'.format(t)
    assert boundedness in ['u', 'sl', 'su', 'b'], 'boundedness must be in [u, sl, su, b]'

    if boundedness in ['sl', 'su', 'b']:
        assert len(bounds) == 2, "when boundedness in ['sl', 'su', 'b'], you must include feasible bounds!"
        assert bounds[1] > bounds[0], "when boundedness in ['sl', 'su', 'b'], you must include feasible bounds!"

    d = y * (1 - y)
    f = y - 0.5
    l = np.log(y / (1 - y))

    # Initiate pdf

    # For the first three terms
    x = a[1] / d

    if a[2] != 0:
        x = x + a[2] * ((f / d) + l)

    # For the fourth term
    if t > 2:
        x = x + a[3]

    # Initalize some counting variables
    e = 1
    o = 1

    # For all other terms greater than 4
    if t > 3:
        for i in range(4, t):
            if i % 2 != 0:
                # iff odd
                x = x + ((o + 1) * a[i] * np.power(f, o))
                o = o + 1

            if i % 2 == 0:
                # iff even
                x = x + a[i] * (((np.power(f, (e + 1))) / d) + (e + 1) * (np.power(f, e)) * l)
                e = e + 1

    # Some change of variables here for boundedness

    x = np.power(x, -1)

    if boundedness != 'u':
        M = quantileMetalog(a, y, t, bounds=bounds, boundedness='u')

    if boundedness == 'sl':
        x = x * np.exp(-M)

    if boundedness == 'su':
        x = x * np.exp(M)

    if boundedness == 'b':
        x = (x * np.power(1 + np.exp(M), 2)) / ((bounds[1] - bounds[0]) * np.exp(M))

    return x


def pdfMetalog_density(m, t, y):
    """
    TODO: write docstring
    """

    # TODO: write assertions
    assert isinstance(m, dict), "m must be a dict including keys ['A', 'params']"
    assert 'A' in m.keys(), "m must be a dict including keys ['A', 'params']"
    assert 'params' in m.keys(), "m must be a dict including keys ['A', 'params']"

    avec = 'a{}'.format(t)
    a = m['A'][avec]
    bounds = m['params']['bounds']
    boundedness = m['params']['boundedness']

    d = y * (1 - y)
    f = y - 0.5
    l = np.log(y / (1 - y))

    # Initiate pdf

    # For the first three terms
    x = a[1] / d

    if a[2] != 0:
        x = x + a[2] * ((f / d) + l)

    # For the fourth term
    if t > 2:
        x = x + a[3]

    # Initalize some counting variables
    e = 1
    o = 1

    # For all other terms greater than 4
    if t > 3:
        for i in range(4, t):
            if i % 2 != 0:
                # iff odd
                x = x + ((o + 1) * a[i] * np.power(f, o))
                o = o + 1

            if i % 2 == 0:
                # iff even
                x = x + a[i] * (((np.power(f, (e + 1))) / d) + (e + 1) * (np.power(f, e)) * l)
                e = e + 1

    # Some change of variables here for boundedness

    x = np.power(x, -1)

    if boundedness != 'u':
        M = quantileMetalog(a, y, t, bounds=bounds, boundedness='u')

    if boundedness == 'sl':
        x = x * np.exp(-M)

    if boundedness == 'su':
        x = x * np.exp(M)

    if boundedness == 'b':
        x = (x * np.power(1 + np.exp(M), 2)) / ((bounds[1] - bounds[0]) * np.exp(M))

    return x

