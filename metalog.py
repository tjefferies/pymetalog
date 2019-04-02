#TODO

import pandas as pd
import numpy as np

from pdf_quantile_functions import pdf_quantile_builder
from a_vector import aVectorOLS


class metalog:

    def __init__(self, x, bounds=(0, 1), boundedness='u', term_limit=13, term_lower_bound=2, step_len=0.01, probs=None, fit_method='any'):
        """
        # TODO: write docstrings


        :param: x: pd.Series
        :param: bounds: list
        :param: boundedness: str
        :param: term_limit: int
        :param: term_lower_bound: int
        :param: step_len: float
        :param: probs: None
        :param: fit_method: str

        """
        # TODO: ask about numpy arrays vs pd.Series input
        # assert isinstance(x, np.ndarray), 'x must be a numpy.array!'
        assert isinstance(x, pd.Series), 'Input x must be pd.Series!'
        assert isinstance(bounds, list), 'Input bounds must be list!'
        assert isinstance(boundedness, str), 'Input boundedness must be str!'

        # TODO: finish assertions

        # Create a dict of dict to hold all the objects
        self.myList = dict()
        self.myList['params'] = dict()
        self.myList['params']['bounds'] = bounds
        self.myList['params']['boundedness'] = boundedness
        self.myList['params']['term_limit'] = term_limit
        self.myList['params']['term_lower_bound'] = term_lower_bound
        self.myList['params']['step_len'] = step_len
        self.myList['params']['fit_method'] = fit_method

        self.x = x
        self.bounds = bounds[:]
        self.term_limit = term_limit
        self.term_lower_bound = term_lower_bound
        self.step_len = step_len
        self.boundedness = boundedness[:]

        if not probs:
            self.MLprobs()
        else:
            self.x = pd.DataFrame(self.x)
            self.x['probs'] = probs[:]

        self.buildZVectorBasedOnBoundedness()
        self.myList['dataValues'] = self.x.copy()
        self.buildYDataFrame()

    def buildZVectorBasedOnBoundedness(self):
        """
        TODO: write docstring
        """
        if self.boundedness == 'u':
            self.x['z'] = self.x[0].copy()
        elif self.boundedness == 'sl':
            self.x['z'] = np.log(self.x[0] - self.bounds[0])
        elif self.boundedness == 'su':
            self.x['z'] = -1.0*np.log(self.bounds[1] - self.x[0])
        elif self.boundedness == 'b':
            self.x['z'] = np.log(self.x[0] - self.bounds[0] / (self.bounds[1] - self.x[0]))

    def buildYDataFrame(self):
        """
        TODO: write docstring
        """
        self.Y = pd.DataFrame(columns=['y1'], data=np.ones(len(self.x)))
        self.Y['y2'] = np.log(self.x['probs'] / (1 - self.x['probs']))
        self.Y['y3'] = (self.x['probs'] - 0.5) * self.Y['y2']
        if self.term_limit > 3:
            self.Y['y4'] = self.x['probs'] - 0.5
        if self.term_limit > 4:
            for i in range(4, self.term_limit):
                y = 'y{}'.format(i)
                if i % 2 != 0:
                    self.Y[y] = np.power(self.Y['y4'].values, i//2)
                if i % 2 == 0:
                    z = 'y{}'.format(i-1)
                    self.Y[y] = np.multiply(self.Y['y2'], self.Y[z])
        self.myList['Y'] = self.Y.copy()

