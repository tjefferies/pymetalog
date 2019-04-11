import pandas as pd
import numpy as np
import scipy as sp
from scipy.optimize import linprog, minimize, NonlinearConstraint
from .pdf_quantile_functions import pdf_quantile_builder
from .support import diffMatMetalog, pdfMetalog, quantileMetalog, newtons_method_metalog

import time

import warnings

def a_vector_OLS_and_LP(m_list,
           term_limit,
           term_lower_bound,
           bounds,
           boundedness,
           fit_method,
           diff_error = .001,
           diff_step = 0.001):

    # Some place holder values
    A = pd.DataFrame()
    c_a_names = []
    c_m_names = []
    Mh = pd.DataFrame()
    Validation = pd.DataFrame()

    # TODO: Large for-loop can probably be factored into smaller functions
    for i in range(term_lower_bound,term_limit+1):
        Y = m_list['Y'].iloc[:,0:i]
        z = m_list['dataValues']['z']
        y = m_list['dataValues']['probs']
        step_len = m_list['params']['step_len']
        methodFit = 'OLS'
        a_name = 'a'+str(i)
        m_name = 'm'+str(i)
        M_name = 'M'+str(i)
        c_m_names = np.append(c_m_names, [m_name, M_name])
        c_a_names = np.append(c_a_names, a_name)

        if fit_method == 'any' or fit_method == 'MLE':
            try:
                temp = np.dot(np.dot(np.linalg.inv(np.dot(Y.T, Y)), Y.T), z)
            except:
                temp = a_vector_LP(m_list, term_limit=i, term_lower_bound=i, diff_error=diff_error, diff_step=diff_step)
                # use LP solver if OLS breaks
        if fit_method == 'OLS':
            try:
                temp = np.dot(np.dot(np.linalg.inv(np.dot(Y.T, Y)), Y.T), z)
            except:
                raise RuntimeError("OLS was unable to solve infeasible or poorly formulated problem")
        if fit_method == "LP":
                temp = a_vector_LP(m_list, term_limit=i, term_lower_bound=i, diff_error=diff_error, diff_step=diff_step)


        temp = np.append(temp, np.repeat(0,(term_limit-i)))

        if fit_method == 'MLE':
            temp = a_vector_MLE(temp, y, i, m_list, bounds, boundedness)
            tempList = pdf_quantile_builder(temp, y=y, term_limit=i, bounds=bounds, boundedness=boundedness)

        # build a y vector for smaller data sets
        if len(z) < 100:
            y2 = np.linspace(step_len, 1 - step_len, ((1 - step_len) / step_len))
            tailstep = step_len / 10
            y1 = np.linspace(tailstep, (min(y2) - tailstep), ((min(y2) - tailstep) / tailstep))
            y3 = np.linspace((max(y2) + tailstep), (max(y2) + tailstep * 9), ((tailstep * 9) / tailstep))
            y = np.hstack((y1, y2, y3))

        # Get the list and quantile values back for validation
        tempList = pdf_quantile_builder(temp, y=y, term_limit=i, bounds=bounds, boundedness=boundedness)

        # If it not a valid pdf run and the OLS version was used the LP version
        if (tempList['valid'] == 'no') and (fit_method != 'OLS'):
            temp = a_vector_LP(m_list, term_limit=i, term_lower_bound=i, diff_error=diff_error, diff_step=diff_step)
            temp = np.append(temp, np.repeat(0, (term_limit - i)))
            methodFit = 'Linear Program'

            # Get the list and quantile values back for validation
            tempList = pdf_quantile_builder(temp, y=y, term_limit=i, bounds=bounds, boundedness=boundedness)



        if len(Mh) != 0:
            Mh = pd.concat([Mh, pd.DataFrame(tempList['m'])], axis=1)
            Mh = pd.concat([Mh, pd.DataFrame(tempList['M'])], axis=1)

        if len(Mh) == 0:
            Mh = pd.DataFrame(tempList['m'])
            Mh = pd.concat([Mh, pd.DataFrame(tempList['M'])], axis=1)

        if len(A) != 0:
            A = pd.concat([A, pd.DataFrame(temp)], axis=1)

        if len(A) == 0:
            A = pd.DataFrame(temp)

        tempValidation = pd.DataFrame(data={'term': [i], 'valid': [tempList['valid']], 'method': [methodFit]})
        Validation = pd.concat([Validation, tempValidation], axis=0)



    A.columns = c_a_names
    Mh.columns = c_m_names

    m_list['A'] = A
    m_list['M'] = Mh
    m_list['M']['y'] = tempList['y']
    m_list['Validation'] = Validation



    return m_list

def a_vector_LP(m_list, term_limit, term_lower_bound, diff_error = .001, diff_step = 0.001):
    cnames = np.array([])

    for i in range(term_lower_bound, term_limit + 1):
        Y = m_list['Y'].iloc[:, 0:i]
        z = m_list['dataValues']['z']

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
            front_zeros = np.repeat(0, (2 * (j - 1)))
            ones = [1, -1]
            trail_zeroes = np.repeat(0, (2 * (len(Y.iloc[:, 1]) - j)))
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
            zeros_temp = np.repeat(0, (2 * len(Y.iloc[:, 0])))

            if np.size(diff_zeros) == 0:
                diff_zeros = zeros_temp
            else:
                diff_zeros = np.vstack((zeros_temp, diff_zeros))

        diff_mat = np.concatenate((diff_zeros, diff_mat), axis=1)

        # Combine the total constraint matrix
        lp_mat = np.concatenate((new, diff_mat), axis=0)

        # Objective function coeficients
        c = np.append(np.repeat(1,(2 * len(Y.iloc[:, 1]))), np.repeat(0, (2*i)))

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


def a_vector_MLE(a, y, term, m_list, bounds, boundedness):
    ym = [newtons_method_metalog(a, xi, term, bounds, boundedness) for xi in m_list['dataValues']['x']]

    def MLE_quantile_constraints(x):
        M = [quantileMetalog(x[:term], yi, term, bounds=bounds, boundedness=boundedness) for yi in x[term:]]
        return m_list['dataValues']['x'] - M

    def MLE_objective_function(x, y, term, m_list):
        return -np.sum([np.log10(pdfMetalog(x[:term], yi, term, bounds, boundedness)) for yi in np.absolute(x[term:])])

    m_list[str('MLE' + str(term))] = {}

    x0 = np.hstack((a[:term],ym))
    m_list[str('MLE' + str(term))]['oldobj'] = -MLE_objective_function(x0, y, term, m_list)
    bnd = ((None, None),)*len(a)+((0, 1),)*(len(x0)-len(a))
    con = NonlinearConstraint(MLE_quantile_constraints, 0, 0)

    mle = minimize(MLE_objective_function, x0, args=(y, term, m_list), bounds=bnd, constraints=con)

    m_list[str('MLE' + str(term))]['newobj'] = -MLE_objective_function(mle.x, y, term, m_list)
    m_list[str('MLE'+str(term))]['A'] = mle.x[:term]
    m_list[str('MLE'+str(term))]['Y'] = mle.x[term:]

    m_list[str('MLE' + str(term))]['oldA'] = a
    m_list[str('MLE' + str(term))]['oldY'] = y

    out_temp = np.zeros_like(a)
    for i in range(term):
        out_temp[i] = mle.x[i]

    return out_temp




