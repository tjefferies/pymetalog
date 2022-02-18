import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from .support import newtons_method_metalog, pdfMetalog_density
from .metalog import metalog


def summary(m):
    """Prints information about the fitted metalog m.
        Prints to console:
          - metalog.output_dict['params']['term_limit']
          - metalog.output_dict['params']['term_lower_bound']
          - metalog.output_dict['params']['boundedness']
          - metalog.output_dict['params']['bounds']
          - metalog.output_dict['params']['step_len']
          - metalog.output_dict['params']['fit_method']
          - metalog.output_dict['Validation']
          - metalog.output_dict['params']['nobs']

    Args:
        m (:obj:`metalog`): A fitted metalog object.

    """
    print(
        " -----------------------------------------------\n",
        "Summary of Metalog Distribution Object\n",
        "-----------------------------------------------\n",
        "\nParameters\n",
        "Term Limit: ",
        m.output_dict["params"]["term_limit"],
        "\n",
        "Term Lower Bound: ",
        m.output_dict["params"]["term_lower_bound"],
        "\n",
        "Boundedness: ",
        m.output_dict["params"]["boundedness"],
        "\n",
        "Bounds (only used based on boundedness): ",
        m.output_dict["params"]["bounds"],
        "\n",
        "Step Length for Distribution Summary: ",
        m.output_dict["params"]["step_len"],
        "\n",
        "Method Use for Fitting: ",
        m.output_dict["params"]["fit_method"],
        "\n",
        "\n\n Validation and Fit Method",
        "Number of Data Points Used: ",
        m.output_dict["params"]["nobs"],
        "\n",
    )
    print(m.output_dict["Validation"].to_string(index=False))


def rmetalog(m, n=1, term=2, generator="rand"):
    """Take n random draws from fitted metalog m using specified number of terms.
        Uses specified random seed.

    Args:
        m (:obj:`metalog`): A fitted metalog object.

        n (:obj:`int`, optional): Number of random draws to take from fitted metalog.
          - strictly >= 1
          - Default: 1

        term (:obj:`int`, optional): Number of metalog terms to use when making random draws.
          - strictly >= 2
          - must be in range [m.term_lower_bound, m.term_limit]
          - Default: 2

        generator (:obj:`str`, optional): String that is used to specify the random number generator.
          - must be in set ('rand','hdr')
            * 'rand' uses `np.random.rand`, results are random each time
            * 'hdr' uses Hubbard Decision Research (HDR) random number generator, results are repeatable
          - Default: 'rand'

    Returns:
        (:obj:`numpy.ndarray`): n length numpy array of random draws from fitted metalog.

    """
    m = m.output_dict
    valid_terms = np.asarray(m["Validation"]["term"])
    valid_terms_printout = " ".join(str(t) for t in valid_terms)

    if (type(n) != int) or (n < 1) or ((n % 1) != 0):
        raise TypeError("Error: n must be a positive numeric interger")
    if (
        (type(term) != int)
        or (term < 2)
        or ((term % 1) != 0)
        or not (term in valid_terms)
    ):
        raise TypeError(
            "Error: term must be a single positive numeric interger contained "
            "in the metalog object. Available terms are: " + valid_terms_printout
        )

    if generator == "hdr":
        x_arr = np.arange(1, n + 1)
        v_index = np.random.randint(80000)

        def hdrgen(pm_index):
            return (
                np.mod(
                    (
                        (
                            np.mod(
                                (v_index + 1000000)
                                ^ 2 + (v_index + 1000000) * (pm_index + 10000000),
                                99999989,
                            )
                        )
                        + 1000007
                    )
                    * (
                        (
                            np.mod(
                                (pm_index + 10000000)
                                ^ 2
                                + (pm_index + 10000000)
                                * (
                                    np.mod(
                                        (v_index + 1000000)
                                        ^ 2
                                        + (v_index + 1000000) * (pm_index + 10000000),
                                        99999989,
                                    )
                                ),
                                99999989,
                            )
                        )
                        + 1000013
                    ),
                    2147483647,
                )
                + 0.5
            ) / 2147483647

        vhdrgen = np.vectorize(hdrgen)
        x = vhdrgen(x_arr)

    else:
        x = np.random.rand(n)

    Y = pd.DataFrame(np.array([np.repeat(1, n)]).T, columns=["y1"])

    # Construct initial Y Matrix values
    Y["y2"] = np.log(x / (1 - x))
    if term > 2:
        Y["y3"] = (x - 0.5) * Y["y2"]
    if term > 3:
        Y["y4"] = x - 0.5

    # Complete the values through the term limit
    if term > 4:
        for i in range(5, (term + 1)):
            y = "".join(["y", str(i)])
            if i % 2 != 0:
                Y[y] = Y["y4"] ** (i // 2)
            if i % 2 == 0:
                z = "".join(["y", str(i - 1)])
                Y[y] = Y["y2"] * Y[z]

    amat = "".join(["a", str(term)])
    a = m["A"][amat].iloc[0:(term)].to_frame()
    s = np.dot(Y, a)

    if m["params"]["boundedness"] == "sl":
        s = m["params"]["bounds"][0] + np.exp(s)

    if m["params"]["boundedness"] == "su":
        s = m["params"]["bounds"][1] - np.exp(-(s))

    if m["params"]["boundedness"] == "b":
        s = (m["params"]["bounds"][0] + (m["params"]["bounds"][1]) * np.exp(s)) / (
            1 + np.exp(s)
        )

    return s


def dmetalog(m, q, term=3):
    """Generate density values with user specified quantiles from a fitted metalog object.
        Generated using user specified number of terms.
        Quantiles are generated using a Newton's Method approximation.

    Args:
        m (:obj:`metalog`): A fitted metalog object.

        q (:obj:`list` | `numpy.ndarray`): Quantiles to return density values for.

        term (:obj:`int`, optional): Number of metalog terms to use when generating densities.
          - strictly >= 2
          - must be in range [m.term_lower_bound, m.term_limit]
          - Default: 3

    Returns:
        (:obj:`list`): len(q) list of density values from fitted metalog.

    """
    valid_terms = np.asarray(m.output_dict["Validation"]["term"])

    if (type(q) != list) and (type(q) != np.ndarray):
        raise TypeError("Error: input q must be a list or numpy array.")

    if (
        (term not in valid_terms)
        or type(term) != int
        or (term < 2)
        or ((term % 1) != 0)
    ):
        raise TypeError(
            "Error: term must be a single positive numeric interger contained in the metalog object. Available "
            "terms are: " + " ".join(map(str, valid_terms))
        )

    qs = list(map(lambda qi: newtons_method_metalog(q=qi, m=m, term=term), q))
    ds = list(map(lambda yi: pdfMetalog_density(y=yi, m=m, t=term), qs))

    return ds


def pmetalog(m, q, term=3):
    """Generate probabilities with user specified quantiles from a fitted metalog object.
        Generated using user specified number of terms.
        Quantiles are generated using a Newton's Method approximation.

    Args:
        m (:obj:`metalog`): A fitted metalog object.

        q (:obj:`list` | `numpy.ndarray`): Quantiles to return probabilities values for.

        term (:obj:`int`, optional): Number of metalog terms to use when generating probabilities.
          - strictly >= 2
          - must be in range [m.term_lower_bound, m.term_limit]
          - Default: 3

    Returns:
        (:obj:`list`): len(q) list of probabilities from fitted metalog.

    """
    valid_terms = np.asarray(m.output_dict["Validation"]["term"])

    if (type(q) != list) and (type(q) != np.ndarray):
        raise TypeError("Error: input q must be a list or numpy array")
    if not isinstance(q, (int, float, complex)) and not all(
        isinstance(x, (int, float, complex)) for x in q
    ):
        raise TypeError("Error: all elements in q must be numeric")
    if (
        (term in valid_terms) != True
        or type(term) != int
        or (term < 2)
        or ((term % 1) != 0)
    ):
        raise TypeError(
            "Error: term must be a single positive numeric interger contained in the metalog object. Available "
            "terms are: " + " ".join(map(str, valid_terms))
        )

    qs = list(map(lambda qi: newtons_method_metalog(q=qi, m=m, term=term), q))
    return qs


def qmetalog(m, y, term=3):
    """Generate quantiles with a probability from a fitted metalog object.

    Args:
        m (:obj:`metalog`): A fitted metalog object.

        y (:obj:`list` | `numpy.ndarray`): Probabilities to return quantile values for.

        term (:obj:`int`, optional): Number of metalog terms to use when generating quantiles.
          - strictly >= 2
          - must be in range [m.term_lower_bound, m.term_limit]
          - Default: 3

    Returns:
        (:obj:`numpy.ndarray`): len(q) length numpy array of quantiles from fitted metalog.

    """
    m = m.output_dict
    valid_terms = np.asarray(m["Validation"]["term"])
    valid_terms_printout = " ".join(str(t) for t in valid_terms)

    if type(y) != list:
        raise TypeError("Error: y must be a list of numeric values")
    y = np.asarray(y)
    if (
        (all(isinstance(x, (int, float, complex)) for x in y)) != True
        or (max(y) >= 1)
        or (min(y) <= 0)
    ):
        raise TypeError(
            "Error: y or all elements in y must be positive numeric values between 0 and 1"
        )
    if (
        (type(term) != int)
        or (term < 2)
        or ((term % 1) != 0)
        or (term in valid_terms) != True
    ):
        raise TypeError(
            "Error: term must be a single positive numeric integer contained "
            "in the metalog object. Available terms are: " + valid_terms_printout
        )

    Y = pd.DataFrame(np.array([np.repeat(1, len(y))]).T, columns=["y1"])

    # Construct the Y Matrix initial values
    Y["y2"] = np.log(y / (1 - y))
    if term > 2:
        Y["y3"] = (y - 0.5) * Y["y2"]
    if term > 3:
        Y["y4"] = y - 0.5

    # Complete the values through the term limit
    if term > 4:
        for i in range(5, (term + 1)):
            y = "".join(["y", str(i)])
            if i % 2 != 0:
                Y[y] = Y["y4"] ** (i // 2)
            if i % 2 == 0:
                z = "".join(["y", str(i - 1)])
                Y[y] = Y["y2"] * Y[z]

    amat = "".join(["a", str(term)])
    a = m["A"][amat].iloc[0:(term)].to_frame()
    s = np.dot(Y, a)

    if m["params"]["boundedness"] == "sl":
        s = m["params"]["bounds"][0] + np.exp(s)

    if m["params"]["boundedness"] == "su":
        s = m["params"]["bounds"][1] - np.exp(-(s))

    if m["params"]["boundedness"] == "b":
        s = (m["params"]["bounds"][0] + (m["params"]["bounds"][1]) * np.exp(s)) / (
            1 + np.exp(s)
        )

    s = s.flatten()
    return s


def plot(m):
    """Plots PDF and Quantile panels for each term of fitted metalog m.

    Args:
        m (:obj:`metalog`): A fitted metalog object.

    Returns:
        (:obj:`dict` with keys ['pdf', 'cdf']): PDF and Quantile panel plots.
    """
    x = m.output_dict
    # build plots
    InitalResults = pd.DataFrame(
        data={
            "term": (
                np.repeat(
                    (str(x["params"]["term_lower_bound"]) + " Terms"),
                    len(x["M"].iloc[:, 0]),
                )
            ),
            "pdfValues": x["M"].iloc[:, 0],
            "quantileValues": x["M"].iloc[:, 1],
            "cumValue": x["M"]["y"],
        }
    )

    if len(x["M"].columns) > 3:
        for i in range(2, ((len(x["M"].iloc[0, :]) - 1) // 2 + 1)):
            TempResults = pd.DataFrame(
                data={
                    "term": np.repeat(
                        (str(x["params"]["term_lower_bound"] + (i - 1)) + " Terms"),
                        len(x["M"].iloc[:, 0]),
                    ),
                    "pdfValues": x["M"].iloc[:, (i * 2 - 2)],
                    "quantileValues": x["M"].iloc[:, (i * 2 - 1)],
                    "cumValue": x["M"]["y"],
                }
            )

            InitalResults = InitalResults.append(
                pd.DataFrame(data=TempResults), ignore_index=True
            )

    # PDF plot
    ymin = np.min(InitalResults["pdfValues"])
    ymax = np.max(InitalResults["pdfValues"])
    nterms = InitalResults.term.nunique()

    nrow = (nterms + 3) // 4

    if nterms < 4:
        ncol = nterms
    else:
        ncol = 4

    pdf_fig, axes = plt.subplots(nrow, ncol, sharey="col", squeeze=False)

    for t in range(nterms):
        data = InitalResults[
            (InitalResults["term"] == (InitalResults.term.unique()[t]))
        ]
        x = data["quantileValues"]
        y = data["pdfValues"]
        r = t // 4
        c = t % 4
        axes[r, c].plot(x, y)
        axes[r, c].set_ylim(ymin, ymax * 1.1)
        axes[r, c].set_title(InitalResults.term.unique()[t])
        axes[r, c].tick_params(axis="both", which="major", labelsize=10)
        axes[r, c].tick_params(axis="both", which="minor", labelsize=10)

    for t in range(nterms, nrow * ncol):
        r = t // 4
        c = t % 4
        axes[r, c].axis("off")

    pdf_fig.text(0.5, 0.04, "Quantile Values", ha="center")
    pdf_fig.text(0.04, 0.5, "PDF Values", va="center", rotation="vertical")

    plt.yscale("linear")
    plt.tight_layout(rect=[0.05, 0.05, 1, 1])

    # Quantile Plot
    ymin = np.min(InitalResults["cumValue"])
    ymax = np.max(InitalResults["cumValue"])
    nterms = InitalResults.term.nunique()

    nrow = (nterms + 3) // 4

    if nterms < 4:
        ncol = nterms
    else:
        ncol = 4

    cdf_fig, axes = plt.subplots(nrow, ncol, sharey="col", squeeze=False)

    for t in range(nterms):
        data = InitalResults[
            (InitalResults["term"] == (InitalResults.term.unique()[t]))
        ]
        x = data["quantileValues"]
        y = data["cumValue"]
        r = t // 4
        c = t % 4
        axes[r, c].plot(x, y)
        axes[r, c].set_ylim(ymin, ymax * 1.1)
        axes[r, c].set_title(InitalResults.term.unique()[t])
        axes[r, c].tick_params(axis="both", which="major", labelsize=10)
        axes[r, c].tick_params(axis="both", which="minor", labelsize=10)

    for t in range(nterms, nrow * ncol):
        r = t // 4
        c = t % 4
        axes[r, c].axis("off")

    cdf_fig.text(0.5, 0.04, "Quantile Values", ha="center")
    cdf_fig.text(0.04, 0.5, "CDF Values", va="center", rotation="vertical")

    plt.yscale("linear")
    plt.tight_layout(rect=[0.05, 0.05, 1, 1])

    return {"pdf": pdf_fig, "cdf": cdf_fig}


def update(m, new_data, penalty=None, alpha=0.0):
    """Updates a previously fitted metalog object with new data.

    Args:
        m (:obj:`metalog`): The previously fitted metalog object to be updated with `new_data`.
          - `save_data` parameter must have been set equal to True in original metalog fit.

        new_data (:obj:`list` | `numpy.ndarray` | `pandas.Series`): Input data to update the metalog object with.
          - must be an array of allowable types: int, float, numpy.int64, numpy.float64

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

    Returns:
        (:obj:`metalog`): Input metalog object that has been updated using `new_data`

    Raises:
      ValueError: 'Input metalog `m.save_data` parameter must be True'
      TypeError: 'Input x must be an array or pandas Series'
      TypeError: 'Input x must be an array of allowable types: int, float, numpy.int64, or numpy.float64'
      IndexError: 'Input x must be of length 3 or greater'
    """

    if not m.save_data:
        raise ValueError("Input metalog `m.save_data` parameter must be True")
    if (
        (type(new_data) != list)
        and (type(new_data) != np.ndarray)
        and (type(new_data) != pd.Series)
    ):
        raise TypeError("Input x must be an array or pandas Series")
    if isinstance(new_data, pd.Series):
        new_data = new_data.values.copy()
    if not all([isinstance(x, (int, float, np.int64, np.float64)) for x in new_data]):
        raise TypeError(
            "Input x must be an array of allowable types: int, float, numpy.int64, or numpy.float64"
        )
    if np.size(new_data) < 3:
        raise IndexError("Input x must be of length 3 or greater")

    old_append_new_data = np.append(m.x, new_data)

    updated_metalog = metalog(
        old_append_new_data,
        bounds=m.output_dict["params"]["bounds"],
        boundedness=m.output_dict["params"]["boundedness"],
        term_limit=m.output_dict["params"]["term_limit"],
        term_lower_bound=m.output_dict["params"]["term_lower_bound"],
        step_len=m.output_dict["params"]["step_len"],
        probs=None,
        fit_method=m.output_dict["params"]["fit_method"],
        penalty=penalty,
        alpha=alpha,
        save_data=True,
    )

    Y = updated_metalog.output_dict["Y"].values
    gamma = Y.T.dot(Y)
    updated_metalog.output_dict["params"]["bayes"]["gamma"] = gamma
    updated_metalog.output_dict["params"]["bayes"]["mu"] = updated_metalog.output_dict[
        "A"
    ]
    v = list()
    for i in range(
        updated_metalog.output_dict["params"]["term_lower_bound"],
        updated_metalog.output_dict["params"]["term_limit"] + 1,
    ):
        v.append(updated_metalog.output_dict["params"]["nobs"] - i)
    v = np.array(v)
    a = v / 2
    updated_metalog.output_dict["params"]["bayes"]["a"] = a
    updated_metalog.output_dict["params"]["bayes"]["v"] = v

    # for now, just using 3 term standard metalog
    v = v[1]
    a = a[1]
    s = np.array([0.1, 0.5, 0.9])
    Ys = np.repeat(1.0, 3)

    Ys = np.column_stack(
        [np.repeat(1, 3), np.log(s / (1 - s)), (s - 0.5) * np.log(s / (1 - s))]
    )
    three_term_metalog_fit_idx = "a{}".format(updated_metalog.term_limit - 3)
    q_bar = np.dot(
        Ys, updated_metalog.output_dict["A"][three_term_metalog_fit_idx].values[-3:]
    )

    updated_metalog.output_dict["params"]["bayes"]["q_bar"] = q_bar

    est = (q_bar[2] - q_bar[1]) / 2 + q_bar[1]
    s2 = ((q_bar[2] - q_bar[1]) / t.ppf(0.9, np.array(v))) ** 2

    gamma = gamma[:3, :3]

    # build covariance matrix for students t
    sig = Ys.dot(np.linalg.solve(gamma, np.eye(len(gamma)))).dot(Ys.T)

    # b = 0.5 * self.output_dict['params']['square_residual_error'][len(self.output_dict['params']['square_residual_error'])]
    b = (a * s2) / gamma[1, 1]
    updated_metalog.output_dict["params"]["bayes"]["sig"] = (b / a) * sig
    updated_metalog.output_dict["params"]["bayes"]["b"] = b

    return updated_metalog
