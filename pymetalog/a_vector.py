import pandas as pd
import numpy as np
import scipy as sp

from scipy.optimize import linprog, minimize, NonlinearConstraint
from .pdf_quantile_functions import pdf_quantile_builder
from .support import diffMatMetalog, pdfMetalog, quantileMetalog, newtons_method_metalog

import time
import warnings

def a_vector_OLS_and_LP(m_dict,
           bounds,
           boundedness,
           term_limit,
           term_lower_bound,
           fit_method,
           alpha,
           diff_error = .001,
           diff_step = 0.001):

    """ Main workhorse function of pymetalog package.
        Called during metalog.__init__ method call.

    Args:
        m_dict (:obj:`dict` with keys ['params', 'dataValues', 'Y']): Initialized output_dict variable from metalog class.
            - m_dict['params']: (:obj:`dict` with keys ['bounds', 'boundedness', 'term_limit', 'term_lower_bound', 'step_len', 'fit_method']):
                * 'bounds': metalog.bounds
                * 'boundedness': metalog.boundedness
                * 'term_limit': metalog.term_limit
                * 'term_lower_bound': metalog.term_lower_bound
                * 'step_len': metalog.step_len
                * 'fit_method': metalog.fit_method

            - m_dict['dataValues']: (:obj:`pandas.DataFrame` with columns ['x','probs','z']  of type numeric):
                * 'x': metalog.x
                * 'probs': metalog.probs
                * 'z': column calculated in metalog.append_zvector method
                    - depends on metalog.boundedness attribute
                    - metalog.boundedness = 'u':
                        * 'z' = metalog.x
                    - metalog.boundedness = 'sl':
                        * 'z' = log( (metalog.x-lower_bound) )
                    - metalog.boundedness = 'su':
                        * 'z' = = log( (upper_bound-metalog.x) )
                    - metalog.boundedness = 'b':
                        * 'z' = log( (metalog.x-lower_bound) / (upper_bound-metalog.x) )

            - m_dict['Y']: (:obj:`pandas.DataFrame` with columns ['y1','y2','y3','y4', ... ,'yn']  of type numeric):
                * 'y1': numpy.array of ones with length equal to len(x)
                * 'y2': numpy.array of numeric values equal to the term attached to s in the logistic quantile function np.log(m_dict['dataValues']['probs'] / (1 - m_dict['dataValues']['probs']))
                * 'y3': numpy.array of numeric values (m_dict['dataValues']['probs'] - 0.5) * m_dict['Y']['y2']
                * 'y4': numpy.array of numeric values m_dict['Y']['y4'] = m_dict['dataValues']['probs'] - 0.5
                * 'yn': numpy.array of numeric values:
                    - if n in 'yn' is odd,
                        m_dict['Y']['yn'] = m_dict['Y']['y4']**(int(i//2))
                    - if n in 'yn' is even,
                        zn = 'y' + str(n-1)
                        m_dict['Y'][yn] = m_dict['Y']['y2'] * m_dict['Y'][zn]

        bounds (:obj:`list`): Upper and lower limits to filter the data with before calculating metalog quantiles/pdfs.
            - should be set in conjunction with the `boundedness` parameter

        boundedness (:obj:`str`): String that is used to specify the type of metalog to fit.
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

        term_limit (:obj:`int`): The upper limit of the range of metalog terms to use to fit the data.
            - strictly > term_lower_bound
            - in range [3,30]

        term_lower_bound (:obj:`int`): The lower limit of the range of metalog terms to use to fit the data.
            - strictly < term_limit
            - in range [2,29]

        fit_method (:obj:`str`): Fit method to use to fit metalog distribution.
            - must be in set ('any','OLS','LP','MLE')
            - Default: 'any'
                * first tries 'OLS' method than 'LP'
            - 'OLS' only tries to fit by solving directly for a coefficients using ordinary least squares method
            - 'LP' only tries to estimate fit using simplex linear program optimization routine
            - 'MLE' first tries 'OLS' method than falls back to a maximum likelihood estimation routine

        alpha (:obj:`float`, optional): Regularization term to add to OLS fit
            - strictly >= 0.
            - should be set in conjunction with `penalty` parameter
            - Default: 0. (no regularization, OLS)

        diff_error (:obj:`float`, optional): Value used to in scipy.optimize.linprog method call
                                             to init the array of values representing the
                                             upper-bound of each inequality constraint (row) in A_ub.
            - #TODO: Insert maths

        diff_step (:obj:`float`, optional): Value passed to `step_len` parameter in support.py diffMatMetalog method call
                                             defines the bin width for the Reimann sum of the differences differentiation method
            - diffMatMetalog differentiates the metalog pdf
                * Differentiation reference: https://math.stackexchange.com/a/313135
    Returns:
        m_dict: (:obj:`dict` with keys ['params', 'dataValues', 'Y', 'A', 'M', 'Validation'])
            - m_dict['A']: (:obj:`pandas.DataFrame` with columns ['a2','a3', ... ,'an'] of type numeric):
                * a2, a3, ... , an are our a coefficients returned by the method specified in `fit_method`

            - m_dict['M']: (:obj:`pandas.DataFrame` with columns 0:'pdf_1',1:'cdf_1',2:'pdf_2',3:'cdf_2',
                            ...,((2*(term_limit-term_lower_bound))+1)-1:'pdf_n',
                                ((2*(term_limit-term_lower_bound))+1):'cdf_n'
                            where n is the total number of metalog fits determined by (term_limit-term_lower_bound)+1
                            )
                * pdf_1, pdf_2, ... , pdf_n are the metalog pdfs returned by pdf_quantile_builder.pdfMetalog method
                * cdf_1, cdf_2, ... , cdf_n are the metalog quantiles returned by pdf_quantile_builder.quantileMetalog method
            
            - m_dict['y']: (:obj: `numpy.ndarray` of type float):
                * Array of bin widths for both the pdf_n and cdf_n

            - m_dict['Validation']: (:obj:`pandas.DataFrame` with columns ['term', 'valid', 'method'] of type str):
                * 'term': each metalog estimation given a number of terms
                * 'valid': boolean flag indicating if the metalog estimation was valid or not
                * 'method': a string indicating which method was used for the metalog estimation

    """

    A = pd.DataFrame()
    c_a_names = []
    c_m_names = []
    Mh = pd.DataFrame()
    Validation = pd.DataFrame()
    df_MH_temp_list = list()
    df_A_temp_list = list()
    df_Validation_temp_list = list()

    # TODO: Large for-loop can probably be factored into smaller functions
    for i in range(term_lower_bound,term_limit+1):
        Y = m_dict['Y'].iloc[:,0:i]
        eye = np.eye(Y.shape[1])
        z = m_dict['dataValues']['z']
        y = m_dict['dataValues']['probs']
        step_len = m_dict['params']['step_len']
        methodFit = 'OLS'
        a_name = 'a'+str(i)
        m_name = 'm'+str(i)
        M_name = 'M'+str(i)
        c_m_names = np.append(c_m_names, [m_name, M_name])
        c_a_names = np.append(c_a_names, a_name)

        if fit_method == 'any' or fit_method == 'MLE':
            try:
                temp = np.dot(np.dot(np.linalg.inv(np.dot(Y.T, Y) + alpha*eye), Y.T), z)
            except:
                # use LP solver if OLS breaks
                temp = a_vector_LP(m_dict, term_limit=i, term_lower_bound=i, diff_error=diff_error, diff_step=diff_step)
                methodFit = 'Linear Program'
        if fit_method == 'OLS':
            try:
                temp = np.dot(np.dot(np.linalg.inv(np.dot(Y.T, Y) + alpha*eye), Y.T), z)
            except:
                raise RuntimeError("OLS was unable to solve infeasible or poorly formulated problem")
        if fit_method == "LP":
            temp = a_vector_LP(m_dict, term_limit=i, term_lower_bound=i, diff_error=diff_error, diff_step=diff_step)
            methodFit = 'Linear Program'

        if fit_method == 'MLE':
            temp = a_vector_MLE(temp, y, i, m_dict, bounds, boundedness)

        temp = np.append(temp, np.zeros(term_limit-i))

        # build a y vector for smaller data sets
        if len(z) < 100:
            y2 = np.linspace(step_len, 1 - step_len, int((1 - step_len) / step_len))
            tailstep = step_len / 10
            y1 = np.linspace(tailstep, (min(y2) - tailstep), int((min(y2) - tailstep) / tailstep))
            y3 = np.linspace((max(y2) + tailstep), (max(y2) + tailstep * 9), int((tailstep * 9) / tailstep))
            y = np.hstack((y1, y2, y3))

        # Get the dict and quantile values back for validation
        temp_dict = pdf_quantile_builder(temp, y=y, term_limit=i, bounds=bounds, boundedness=boundedness)

        # If it not a valid pdf run and the OLS version was used the LP version
        if (temp_dict['valid'] == 'no') and (fit_method != 'OLS'):
            temp = a_vector_LP(m_dict, term_limit=i, term_lower_bound=i, diff_error=diff_error, diff_step=diff_step)
            temp = np.append(temp, np.zeros(term_limit-i))
            methodFit = 'Linear Program'

            # Get the dict and quantile values back for validation
            temp_dict = pdf_quantile_builder(temp, y=y, term_limit=i, bounds=bounds, boundedness=boundedness)

        df_MH_temp_list.append(pd.DataFrame(temp_dict['m']))
        df_MH_temp_list.append(pd.DataFrame(temp_dict['M']))
        df_A_temp_list.append(pd.DataFrame(temp))

        tempValidation = pd.DataFrame(data={'term': [i], 'valid': [temp_dict['valid']], 'method': [methodFit]})
        df_Validation_temp_list.append(tempValidation)

    Validation = pd.concat(df_Validation_temp_list, axis=0)
    Mh = pd.concat(df_MH_temp_list, axis=1)
    A = pd.concat(df_A_temp_list, axis=1)

    A.columns = c_a_names
    Mh.columns = c_m_names

    m_dict['A'] = A
    m_dict['M'] = Mh
    m_dict['M']['y'] = temp_dict['y']
    m_dict['Validation'] = Validation

    A = np.column_stack((np.repeat(1.,len(A)), A))
    Est = np.dot(m_dict['Y'], A)
    ncols = A.shape[1]
    Z = np.column_stack((np.array(m_dict['dataValues']['z']),np.repeat(m_dict['dataValues']['z'].values,ncols-1).reshape(len(m_dict['dataValues']['z']),ncols-1)))

    m_dict['square_residual_error'] = ((Z-Est)**2).sum(axis=1)

    return m_dict

def a_vector_LP(m_dict, term_limit, term_lower_bound, diff_error = .001, diff_step = 0.001):
    """TODO: write docstring

    """
    cnames = np.array([])

    for i in range(term_lower_bound, term_limit + 1):
        Y = m_dict['Y'].iloc[:, 0:i]
        z = m_dict['dataValues']['z']

        # Bulding the objective function using abs value LP formulation
        Y_neg = -Y

        new_Y = pd.DataFrame({'y1': Y.iloc[:, 0], 'y1_neg': Y_neg.iloc[:, 0]})

        for c in range(1,len(Y.iloc[0,:])):
            new_Y['y'+str(c+1)] = Y.iloc[:,c]
            new_Y['y' + str(c+1)+'_neg'] = Y_neg.iloc[:, c]

        a = np.array([''.join(['a', str(i)])])
        cnames = np.append(cnames, a, axis=0)

        # Building the constraint matrix
        error_mat = np.array([])

        for j in range(1,len(Y.iloc[:,0])+1):
            front_zeros = np.zeros(2 * (j - 1))
            ones = [1, -1]
            trail_zeroes = np.zeros(2 * (len(Y.iloc[:, 1]) - j))
            if j == 1:
                error_vars = np.append(ones, trail_zeroes)

            elif j != 1:
                error_vars = np.append(front_zeros, ones)
                error_vars = np.append(error_vars, trail_zeroes)

            if error_mat.size == 0:
                error_mat = np.append(error_mat, error_vars, axis=0)
            else:
                error_mat = np.vstack((error_mat, error_vars))

        new = pd.concat((pd.DataFrame(data=error_mat), new_Y), axis=1)
        diff_mat = diffMatMetalog(i, diff_step)
        diff_zeros = []

        for t in range(0,len(diff_mat.iloc[:, 0])):
            zeros_temp = np.zeros(2 * len(Y.iloc[:, 0]))

            if np.size(diff_zeros) == 0:
                diff_zeros = zeros_temp
            else:
                diff_zeros = np.vstack((zeros_temp, diff_zeros))

        diff_mat = np.concatenate((diff_zeros, diff_mat), axis=1)

        # Combine the total constraint matrix
        lp_mat = np.concatenate((new, diff_mat), axis=0)

        # Objective function coeficients
        c = np.append(np.ones(2 * len(Y.iloc[:, 1])), np.zeros(2*i))

        # Constraint matrices
        A_eq = lp_mat[:len(Y.iloc[:, 1]),:]
        A_ub = -1*lp_mat[len(Y.iloc[:, 1]):,:]
        b_eq = z
        b_ub = -1*np.repeat(diff_error, len(diff_mat[:,0]))

        # Solving the linear program w/ scipy (for now)
        lp_sol = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='simplex', options={"maxiter":5000, "tol":1.0e-5,"disp": False})

        # Consolidating solution back into the a vector
        tempLP = lp_sol.x[(2 * len(Y.iloc[:, 1])):(len(lp_sol.x)+1)]
        temp = []

        for r in range(0,((len(tempLP) // 2))):
            temp.append(tempLP[(r * 2)] - tempLP[(2 * r)+1])

    return temp


def a_vector_MLE(a, y, term, m_dict, bounds, boundedness):
    """TODO: write docstring

    """
    ym = [newtons_method_metalog(a, xi, term, bounds, boundedness) for xi in m_dict['dataValues']['x']]

    def MLE_quantile_constraints(x):
        M = [quantileMetalog(x[:term], yi, term, bounds=bounds, boundedness=boundedness) for yi in x[term:]]
        return m_dict['dataValues']['x'] - M

    def MLE_objective_function(x, y, term, m_dict):
        return -np.sum([np.log10(pdfMetalog(x[:term], yi, term, bounds, boundedness)) for yi in np.absolute(x[term:])])

    m_dict[str('MLE' + str(term))] = {}

    x0 = np.hstack((a[:term],ym))
    m_dict[str('MLE' + str(term))]['oldobj'] = -MLE_objective_function(x0, y, term, m_dict)
    bnd = ((None, None),)*len(a)+((0, 1),)*(len(x0)-len(a))
    con = NonlinearConstraint(MLE_quantile_constraints, 0, 0)

    mle = minimize(MLE_objective_function, x0, args=(y, term, m_dict), bounds=bnd, constraints=con)

    m_dict[str('MLE' + str(term))]['newobj'] = -MLE_objective_function(mle.x, y, term, m_dict)
    m_dict[str('MLE'+str(term))]['A'] = mle.x[:term]
    m_dict[str('MLE'+str(term))]['Y'] = mle.x[term:]

    m_dict[str('MLE' + str(term))]['oldA'] = a
    m_dict[str('MLE' + str(term))]['oldY'] = y

    out_temp = np.zeros_like(a)
    for i in range(term):
        out_temp[i] = mle.x[i]

    return out_temp




