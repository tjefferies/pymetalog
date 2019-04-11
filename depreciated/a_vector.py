import pandas as pd
import numpy as np
from copy import deepcopy
from pdf_quantile_functions import pdf_quantile_builder


class aVectorOLS:

    def __init__(self,
                 myList,
                 term_limit,
                 term_lower_bound,
                 bounds,
                 boundedness,
                 fit_method,
                 diff_error=.001,
                 diff_step=0.001):
        """

        :param: myList: dict
        :param: term_limit: int
        :param: term_lower_bound: int
        :param: bounds: list
        :param: boundedness: list
        :param: fit_method: list
        :param: diff_error: float
        :param: diff_step: float

        """
        self.term_limit = term_limit
        self.term_lower_bound = term_lower_bound
        self.myList = deepcopy(myList)
        self.A = pd.DataFrame()
        self.c_a_names = list()
        self.c_m_names = list()
        self.Mh = pd.DataFrame()
        self.Validation = pd.DataFrame()
        self.diff_step = diff_step

        # TODO: finish assertions

    def run(self):
        """
        TODO: write docstring
        """
        # TODO: Large for-loop can probably be factored into smaller functions
        for i in range(self.term_lower_bound, self.term_limit):
            Y = self.myList['Y'].iloc[:, 0:i + 1].values
            z = self.myList['dataValues']['z'].values
            y = self.myList['dataValues']['probs']
            step_len = self.myList['params']['step_len']
            methodFit = 'OLS'
            a = 'a{}'.format(i)
            m_name = 'm{}'.format(i)
            M_name = 'M{}'.format(i)
            self.c_m_names = self.c_m_names + [m_name, M_name]
            self.c_a_names = self.c_a_names + [a]

            # Try to use the OLS approach
            try:
                temp = np.linalg.lstsq(Y, z, rcond=None)[0]
            except:
                pass

            # Use LP if temp is not a valid numeric vector
            #             if not isinstance(temp, np.ndarray):
            # TODO: finish R function a_vector_OLS_and_LP

            temp = temp + np.zeros(self.term_limit - i).tolist()

            # build a y vector for smaller data sets

            if len(z) < 100:
                y = np.arange(self.step_len, 1, self.step_len).tolist()
                tailstep = self.step_len / 10
                y = np.arange(tailstep, (min(y) - tailstep), tailstep).tolist() + y + \
                     np.arange(max(y) + tailstep, max(y) + tailstep * 9, tailstep).tolist()

            # Get the list and quantile values back for validation

            tempList = pdf_quantile_builder(
                temp,
                y=y,
                term_limit=i,
                bounds=bounds,
                boundedness=boundedness
            )

            if len(self.Mh) != 0:
                self.Mh = pd.concat([self.Mh, pd.Series(tempList['m'])], axis=1)
                self.Mh = pd.concat([self.Mh, pd.Series(tempList['M'])], axis=1)

            if len(self.Mh) == 0:
                self.Mh = pd.DataFrame(tempList['m'])
                self.Mh = pd.concat([self.Mh, pd.Series(tempList['M'])], axis=1)

            if len(self.A) != 0:
                self.A = pd.concat([self.A, pd.Series(temp)], axis=1)

            if len(self.A) == 0:
                self.A = pd.DataFrame(temp)

            tempValidation = pd.DataFrame([{'term': i, 'valid': tempList['valid'], 'method': methodFit}])

            self.Validation = self.Validation.append(tempValidation)

        self.A.columns = self.c_a_names
        self.Mh.columns = self.c_m_names
        self.myList['A'] = self.A
        self.myList['M'] = self.Mh
        self.myList['M']['y'] = tempList['y']
        self.myList['Validation'] = self.Validation.copy()

    def a_vector_LP(self):
        """
        TODO: write docstring
        TODO: Finish function
        """
        A = pd.DataFrame()
        cnames = list()
        for i in range(self.term_lower_bound, self.term_limit):
            Y = self.myList['Y'].iloc[:, 0:i + 1].values
            z = self.myList['dataValues']['z'].values
            Y_neg = -Y
            new_Y = np.transpose(np.array([Y.iloc[:, 0], Y_neg.iloc[:, 0]]))
            for c in range(1, len(Y[0, :])):
                new_Y = pd.concat([pd.Series(new_Y), pd.Series(Y[:, c])], axis=1).values
                new_Y = pd.concat([pd.DataFrame(new_Y), pd.Series(Y_neg[:, c])], axis=1).values
            a = 'a{}'.format(i)
            cnames.append(a)

            # Building the constraint matrix
            error_mat = np.array()

            for j in range(1, len(Y) + 1):
                front_zeros = np.zeros(2 * (j - 1))
                ones = [1, -1]
                trail_zeroes = np.zeros(2 * (len(Y[:, 0]) - j))

                if j == 1:
                    error_vars = [ones, trail_zeroes]
                else:
                    error_vars = [front_zeros, ones, trail_zeroes]

                error_mat = np.vstack(np.array(cnames), np.array(error_vars))

            new = pd.concat([pd.Series(error_mat), pd.Series(new_Y)], axis=1).values
            diff_mat = diffMatMetalog(i, self.diff_step)
            diff_zeros = list()

            for t in range(len(diff_mat)):
                zeros_temp = np.zeros(2 * len(Y))
                diff_zeros.append(zeros_temp)

            diff_mat = pd.concat([pd.Series(diff_zeros), diff_mat], axis=1).values

            prob = LpProblem('a{} Problem'.format(i), LpMinimize)

            # Combine the total constraint matrix
            lp_mat = pd.concat([pd.DataFrame(new), pd.DataFrame(diff_mat)]).values

            # Objective function coeficients
            f_obj = [np.ones(2 * len(Y)), np.zeros(2 * i)]

            # Constraint matrix
            f.con = lp_mat

