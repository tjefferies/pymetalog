import numpy as np
import pandas as pd
from .support import MLprobs
from .a_vector import a_vector_OLS_and_LP


class metalog():
    """
    Main class in pymetalog package.
    The pymetalog package is a python implementation of Tom Keelin's metalog distributions.

    The metalog distributions are a family of continuous univariate
    probability distributions that are convenient derivations of equations that appropriately
    characterize the probability density (PDF), cumulative (CDF) or quantile distribution functions.

    They can be used in most any situation in which CDF data is known and a flexible,
    simple, and easy-to-use continuous probability distribution is needed to
    represent that data. See links below for more details.

    Uses and benefits: http://www.metalogdistributions.com/usesbenefits.html
    Applications: http://www.metalogdistributions.com/applicationsdata.html

    Theory: http://pubsonline.informs.org/doi/abs/10.1287/deca.2016.0338
    Homepage: http://www.metalogdistributions.com/

    Attributes:
        x (:obj: `numpy.ndarray`): Input array to fit the metalog distribution to.
        boundedness (:obj: `str`): String type of metalog to fit ('u' | 'sl' | 'su' | 'b').
        bounds (:obj: `list`): List upper and lower limits to filter array with before calculating metalog quantiles/pdfs.
        term_limit (:obj: `int`): Int upper limit of the range of metalog terms to use to fit the data.
        term_lower_bound (:obj: `int`): Int lower limit of the range of metalog terms to use to fit the data.
        step_len (:obj: `float`): Float bin width used to estimate the metalog fit.
        probs (:obj: `numpy.ndarray`): Input array of probabilities associated with the data values in `x`.
        fit_method (:obj: `str`): String type of metalog fit method ('any' | 'OLS' | 'LP' | 'MLE').
        penalty (:obj:`str`, optional): Used to specify the norm used in the regularization.
        alpha (:obj:`float`, optional): Regularization term to add to OLS fit.

        output_dict (:obj:`dict` with keys ['params', 'dataValues', 'Y', 'A', 'M', 'Validation']).
            - output_dict['params'] (:obj:`dict`):
                - output_dict['params']['bounds'] = `bounds`
                - output_dict['params']['boundedness'] = `boundedness`
                - output_dict['params']['term_limit'] = `term_limit`
                - output_dict['params']['term_lower_bound'] = `term_lower_bound`
                - output_dict['params']['step_len'] = `step_len`
                - output_dict['params']['fit_method'] = `fit_method`

            - output_dict['dataValues'] (:obj:`dict`).
                - output_dict['dataValues']['x']: `x`
                - output_dict['dataValues']['probs']: `probs`
                - output_dict['dataValues']['z']: column calculated in `append_zvector` method
                    * depends on `boundedness` attribute
                    * `boundedness` = 'u':
                        * output_dict['dataValues']['z'] = `x`
                    * `boundedness` = 'sl':
                        * output_dict['dataValues']['z'] = log( (`x`-lower_bound) )
                    * `boundedness` = 'su':
                        * output_dict['dataValues']['z'] = = log( (upper_bound-`x`) )
                    * `boundedness` = 'b':
                        * output_dict['dataValues']['z'] = log( (`x`-lower_bound) / (upper_bound-`x`) )

            - output_dict['Y'] (:obj:`pandas.DataFrame` with columns ['y1','y2','y3','y4', ... ,'yn']  of type numeric).
                - output_dict['Y']['y1']: numpy.array of ones with length equal to len(`x`)
                - output_dict['Y']['y2']: numpy.array of numeric values equal to the term attached to s in the logistic quantile function np.log(output_dict['dataValues']['probs'] / (1 - output_dict['dataValues']['probs']))
                - output_dict['Y']['y3']: numpy.array of numeric values (output_dict['dataValues']['probs'] - 0.5) * output_dict['Y']['y2']
                - output_dict['Y']['y4']: numpy.array of numeric values output_dict['Y']['y4'] = output_dict['dataValues']['probs'] - 0.5
                - output_dict['Y']['yn']: numpy.array of numeric values:
                    * if n in 'yn' is odd,
                        output_dict['Y']['yn'] = output_dict['Y']['y4']**(int(i//2))
                    * if n in 'yn' is even,
                        zn = 'y' + str(n-1)
                        output_dict['Y'][yn] = output_dict['Y']['y2'] * output_dict['Y'][zn]

            - output_dict['A']: (:obj:`pandas.DataFrame` with columns ['a2','a3', ... ,'an'] of type numeric):
                * 'a2', 'a3', ... , 'an' are our a coefficients returned by the method specified in `fit_method`

            - output_dict['M']: (:obj:`pandas.DataFrame` with columns ['m2', 'M2', 'm3', 'M3', ... , 'mn', 'Mn'] of type numeric):
                * 'm2', 'M2', 'm3', 'M3', ... , 'mn', 'Mn' are the metalog pdf/quantile fit estimates returned by the method specified in `fit_method`
                * 'mn' is the pdf fit of metalog term n
                * 'Mn' is the quantile fit of metalog term n

            - output_dict['Validation']: (:obj:`pandas.DataFrame` with columns ['term', 'valid', 'method']):
                * 'term' (:obj: `int`): each metalog estimation given a number of terms
                * 'valid' (:obj: `str`): boolean flag indicating if the metalog estimation was valid or not
                * 'method' (:obj: `str`): a string indicating which method was used for the metalog estimation

    Methods:
        get_params(`bounds`, `boundedness`, `term_limit`, `term_lower_bound`, `step_len`, `fit_method`) -> output_dict['params'] (:obj:`dict`)
        append_zvector(`bounds`, `boundedness`) -> df_x: (:obj:`pandas.DataFrame` with columns ['x','probs','z'] of type numeric)
        
    """
    def __init__(self, x, bounds=[0,1], boundedness='u', term_limit=13, term_lower_bound=2, step_len=.01, probs=None, fit_method='any', penalty=None, alpha=0.):
        """Fits a metalog distribution using the input array `x`.

        Args:
            x (:obj:`list` | `numpy.ndarray` | `pandas.Series`): Input data to fit the metalog distribution to.
                - must be an array of allowable types: int, float, numpy.int64, numpy.float64

            bounds (:obj:`list`, optional): Upper and lower limits to filter the data with before calculating metalog quantiles/pdfs.
                - should be set in conjunction with the `boundedness` parameter
                - Default: [0,1]

            boundedness (:obj:`str`, optional): String that is used to specify the type of metalog to fit.
                - must be in set ('u','sl','su','b')
                - Default: 'u'
                    * Fits an unbounded metalog
                - 'sl' fits a strictly lower bounded metalog
                    * len(bounds) must == 1
                - 'su' fits a strictly upper bounded metalog
                    * len(bounds) must == 1
                - 'b' fits a upper/lower bounded metalog
                    * len(bounds) must == 2
                    * bounds[1] must be > bounds[0]

            term_limit (:obj:`int`, optional): The upper limit of the range of metalog terms to use to fit the data.
                - strictly > term_lower_bound
                - in range [3,30]

            term_lower_bound (:obj:`int`, optional): The lower limit of the range of metalog terms to use to fit the data.
                - strictly < term_limit
                - in range [2,29]

            step_len (:obj:`float`, optional): Used to specify the bin width used to estimate the metalog.
                - must be in range [0.001, 0.01]

            probs (:obj:`list` | `numpy.ndarray`, optional): Probabilities associated with the data values in x.
                - must be an array of integer or float data
                - all elements must be in range [0,1]

            fit_method (:obj:`str`, optional): Fit method to use to fit metalog distribution.
                - must be in set ('any','OLS','LP','MLE')
                - Default: 'any'
                    * first tries 'OLS' method than 'LP'
                - 'OLS' only tries to fit by solving directly for a coefficients using ordinary least squares method
                - 'LP' only tries to estimate fit using simplex linear program optimization routine
                - 'MLE' first tries 'OLS' method than falls back to a maximum likelihood estimation routine

            penalty (:obj:`str`, optional): Used to specify the norm used in the regularization.
                - must be in set ('l2', None)
                    * 'l2' performs Ridge Regression instead of OLS
                        - Automatically shrinks a coefficients, leading to "smoother" fits
                - should be set in conjunction with `alpha` parameter
                - Default: None

            alpha (:obj:`float`, optional): Regularization term to add to OLS fit.
                - strictly >= 0.
                - should be set in conjunction with `penalty` parameter
                - Default: 0. (no regularization, OLS)

        Raises:
            TypeError: 'Input x must be an array or pandas Series'
            TypeError: 'Input x must be an array of allowable types: int, float, numpy.int64, or numpy.float64'
            TypeError: 'bounds parameter must be of type list'
            TypeError: 'bounds parameter must be list of integers'
            TypeError: 'term_limit parameter should be an integer between 3 and 30'
            TypeError: 'term_lower_bound parameter should be an integer'
            TypeError: 'Input probabilities must be an array'
            TypeError: 'Input probabilities must be an array of integer or float data'

            IndexError: 'Input x must be of length 3 or greater'
            IndexError: 'Must supply only one bound for semi-lower or semi-upper boundedness'
            IndexError: 'Must supply exactly two bounds for bounded boundedness (i.e. [0,30])'
            IndexError: 'probs vector and x vector must be the same length'

            ValueError: 'for semi-lower boundedness the lower bound must be less than the smallest value in x'
            ValueError: 'for semi-upper boundedness the upper bound must be greater than the largest value in x'
            ValueError: 'Upper bound must be greater than lower bound'
            ValueError: 'boundedness parameter must be u, su, sl or b only'
            ValueError: 'term_limit parameter should be an integer between 3 and 30'
            ValueError: 'term_limit must be less than or equal to the length of the vector x'
            ValueError: 'term_lower_bound parameter should be greater than or equal to 2'
            ValueError: 'term_lower_bound parameter must be less than or equal to term_limit parameter'
            ValueError: 'step_len must be >= to 0.001 and <= to 0.01'
            ValueError: 'Input probabilities cannot contain nans'
            ValueError: 'Input probabilities must have values between, not including, 0 and 1'
            ValueError: 'fit_method can only be values OLS, LP, any, or MLE'
            ValueError: 'penalty can only be values l2 or None'
            ValueError: 'alpha must only be a float >= 0.'

        Example:

            Fit a metalog to a numpy.ndarray of numeric data.
            
            >>> import numpy as np
                import pandas as pd
                import matplotlib.pyplot as plt
                import pymetalog as pm

            >>> fish_data = np.loadtxt('fishout.csv', delimiter=',', skiprows=1, dtype='str')[:,1].astype(np.float)
            >>> fish_metalog = pm.metalog(x=fish_data, bounds=[0,60], boundedness='b', term_limit=9, term_lower_bound=2, step_len=.001,)
            >>> pm.summary(fish_metalog)
            >>> # plot function - right now this saves plots to local
                pm.plot(fish_metalog)
                plt.show()

        """

        self.x = x.copy()
        self.boundedness = boundedness
        self.bounds = bounds[:]
        self.term_limit = term_limit
        self.term_lower_bound = term_lower_bound
        self.step_len = step_len
        self.probs = probs
        self.fit_method = fit_method
        self.penalty = penalty

        if penalty == None:
            alpha = 0.

        self.alpha = alpha

        if probs == None:
            df_x = MLprobs(self.x, step_len=step_len)

        else:
            df_x = pd.DataFrame()
            df_x['x'] = self.x
            df_x['probs'] = self.probs

        output_dict = {}

        # build z vector based on boundedness
        df_x = self.append_zvector(df_x)

        output_dict['params'] = self.get_params()
        output_dict['dataValues'] = df_x

        # Construct the Y Matrix initial values
        Y = pd.DataFrame()
        Y['y1'] = np.ones(len(df_x['x']))
        Y['y2'] = np.log(df_x['probs'] / (1 - df_x['probs']))
        Y['y3'] = (df_x['probs'] - 0.5) * Y['y2']

        if self.term_limit > 3:
            Y['y4'] = df_x['probs'] - 0.5

         # Complete the values through the term limit
        if (term_limit > 4):
            for i in range(5, self.term_limit+1):
                yn = 'y'+str(i)

                if i % 2 != 0:
                    Y[yn] = Y['y4']**(int(i//2))

                if i % 2 == 0:
                    zn = 'y' + str(i-1)
                    Y[yn] = Y['y2'] * Y[zn]

        output_dict['Y'] = Y

        self.output_dict = a_vector_OLS_and_LP(
            output_dict,
            bounds = self.bounds,
            boundedness = self.boundedness,
            term_limit = self.term_limit,
            term_lower_bound = self.term_lower_bound,
            fit_method = self.fit_method,
            alpha = self.alpha,
            diff_error = .001,
            diff_step = 0.001)


    # input validation...
    @property
    def x(self):
        """x (:obj:`list` | `numpy.ndarray` | `pandas.Series`): Input data to fit a metalog to."""
        
        return self._x

    @x.setter
    def x(self, xs):
        if (type(xs) != list) and (type(xs) != np.ndarray) and (type(xs) != pd.Series):
            raise TypeError('Input x must be an array or pandas Series')
        if isinstance(xs, pd.Series):
            xs = xs.values.copy()
        if not all(isinstance(x, (int, float, np.int64, np.float64)) for x in xs):
            raise TypeError('Input x must be an array of allowable types: int, float, numpy.int64, or numpy.float64')
        if np.size(xs) < 3:
            raise IndexError('Input x must be of length 3 or greater')
        self._x = xs

    @property
    def bounds(self):
        """bounds (:obj:`list`, optional): Upper and lower limits to filter the data with before calculating metalog quantiles/pdfs."""
        
        return self._bounds

    @bounds.setter
    def bounds(self, bs):
        if type(bs) != list:
            raise TypeError('bounds parameter must be of type list')
        if not all(isinstance(x, (int)) for x in bs):
            raise TypeError('bounds parameter must be list of integers')
        if (self.boundedness == 'sl' or self.boundedness == 'su') and len(bs) != 1:
            raise IndexError('Must supply only one bound for semi-lower or semi-upper boundedness')
        if self.boundedness == 'b' and len(bs) != 2:
            raise IndexError('Must supply exactly two bounds for bounded boundedness (i.e. [0,30])')
        if self.boundedness == 'su':
            bs_o = [np.min(self.x), bs[0]]
        if self.boundedness == 'sl':
            bs_o = [bs[0], np.max(self.x)]
        if self.boundedness == 'b' or self.boundedness == 'u':
            bs_o = bs
        if self.boundedness == 'sl' and np.min(self.x) < bs_o[0]:
            raise ValueError('for semi-lower boundedness the lower bound must be less than the smallest value in x')
        if self.boundedness == 'su' and np.max(self.x) > bs_o[1]:
            raise ValueError('for semi-upper boundedness the upper bound must be greater than the largest value in x')
        if bs_o[0] > bs_o[1] and self.boundedness == 'b':
            raise ValueError('Upper bound must be greater than lower bound')
        self._bounds = bs_o

    @property
    def boundedness(self):
        """boundedness (:obj:`str`, optional): String that is used to specify the type of metalog to fit."""
        
        return self._boundedness

    @boundedness.setter
    def boundedness(self, bns):
        if bns != 'u' and bns != 'b' and bns != 'su' and bns != 'sl':
            raise ValueError('boundedness parameter must be u, su, sl or b only')
        self._boundedness = bns

    @property
    def term_limit(self):
        """term_limit (:obj:`int`, optional): The upper limit of the range of a coefficients to generate."""
        
        return self._term_limit

    @term_limit.setter
    def term_limit(self, tl):
        if type(tl) != int:
            raise TypeError('term_limit parameter should be an integer between 3 and 30')
        if tl > 30 or tl < 3:
            raise ValueError('term_limit parameter should be an integer between 3 and 30')
        if tl > len(self.x):
            raise ValueError('term_limit must be less than or equal to the length of the vector x')
        self._term_limit = tl

    @property
    def term_lower_bound(self):
        """term_lower_bound (:obj:`int`, optional): The lower limit of the range of a coefficients to generate."""
        
        return self._term_lower_bound

    @term_lower_bound.setter
    def term_lower_bound(self, tlb):
        if type(tlb) != int:
            raise TypeError('term_lower_bound parameter should be an integer')
        if tlb < 2:
            raise ValueError('term_lower_bound parameter should be greater than or equal to 2')
        if tlb > self.term_limit:
            raise ValueError('term_lower_bound parameter must be less than or equal to term_limit parameter')
        self._term_lower_bound = tlb

    @property
    def step_len(self):
        """step_len (:obj:`float`, optional): Used to specify the bin width used to estimate the metalog."""
        
        return self._step_len

    @step_len.setter
    def step_len(self, sl):
        if sl < .001 or sl > .01:
            raise ValueError('step_len must be >= to 0.001 and <= to 0.01')
        self._step_len = sl

    @property
    def probs(self):
        """probs (:obj:`list` | `numpy.ndarray`, optional): Probabilities associated with the data values in x."""
        
        return self._probs

    @probs.setter
    def probs(self, ps):
        if ps != None:
            if not isinstance(ps, (list, np.ndarray)):
                raise TypeError('Input probabilities must be an array')
            if not all(isinstance(x, (int, float)) for x in ps):
                raise TypeError('Input probabilities must be an array of integer or float data')
            if np.size(np.where(np.isnan(ps))) != 0:
                raise ValueError('Input probabilities cannot contain nans')
            if np.max(ps) > 1 or np.min(ps) < 0:
                raise ValueError('Input probabilities must have values between, not including, 0 and 1')
            if len(ps) != len(self.x):
                raise IndexError('probs vector and x vector must be the same length')
            ps = ps.copy()
        self._probs = ps

    @property
    def fit_method(self):
        """fit_method (:obj:`str`, optional): Fit method to use to fit metalog distribution."""

        return self._fit_method

    @fit_method.setter
    def fit_method(self, fm):
        if fm != 'OLS' and fm != 'LP' and fm != 'any' and fm != 'MLE':
            raise ValueError('fit_method can only be values OLS, LP, any, or MLE')
        self._fit_method = fm

    @property
    def penalty(self):
        """penalty (:obj:`str`, optional): Used to specify the norm used in the regularization."""

        return self._penalty

    @fit_method.setter
    def penalty(self, p):
        if p != 'l2' and p is not None:
            raise ValueError('penalty can only be values l2 or None')
        self._penalty = p

    @property
    def alpha(self):
        """alpha (:obj:`float`): L2 regularization term to add to OLS fit"""

        return self._alpha

    @alpha.setter
    def alpha(self, a):
        if a < 0 or not isinstance(a,float):
            raise ValueError('alpha must only be a float >= 0.')
        self._alpha = a

    def get_params(self):
        """Sets the `params` key (dict) of `output_dict` object prior to input to `a_vector_OLS_and_LP` method.
            - Uses metalog attributes to set keys

        Returns:
            params: (:obj:`dict`): Dictionary that is used as input to `a_vector_OLS_and_LP` method.

        """

        params = {}
        params['bounds'] = self.bounds
        params['boundedness'] = self.boundedness
        params['term_limit'] = self.term_limit
        params['term_lower_bound'] = self.term_lower_bound
        params['step_len'] = self.step_len
        params['fit_method'] = self.fit_method

        return params

    def append_zvector(self, df_x):
        """Sets the `dataValues` key (pandas.DataFrame) of `output_dict` object prior to input to `a_vector_OLS_and_LP` method.
            
        Uses `boundedness` attribute to set z vector
            - 'u': output_dict['dataValues']['z'] = x
                * Start with all the input data
            - 'sl': output_dict['dataValues']['z'] = log( (x-lower_bound) )
            - 'su': output_dict['dataValues']['z'] = log( (upper_bound-x) )
            - 'b': output_dict['dataValues']['z'] = log( (x-lower_bound) / (upper_bound-x) )

        Returns:
            df_x: (:obj:`pandas.DataFrame` with columns ['x','probs','z'] of type numeric): DataFrame that is used as input to `a_vector_OLS_and_LP` method.
                - df_x['x']: metalog.x
                - df_x['probs']: metalog.probs
                - df_x['z']: z vector above
        """

        if self.boundedness == 'u':
            df_x['z'] = df_x['x']
        if self.boundedness == 'sl':
            df_x['z'] = np.log(np.array((df_x['x'] - self.bounds[0]), dtype=np.float64))
        if self.boundedness == 'su':
            df_x['z'] = -np.log(np.array((self.bounds[1]-df_x['x']), dtype=np.float64))
        if self.boundedness == 'b':
            df_x['z'] = np.log(np.array(((df_x['x'] - self.bounds[0])/(self.bounds[1]-df_x['x'])), dtype=np.float64))

        return df_x

    def __getitem__(self):
        return self.output_dict

    def __getitem__(self, arr):
        if arr not in self.output_dict:
            raise KeyError()
        return self.output_dict[arr]

