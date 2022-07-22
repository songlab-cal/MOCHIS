import gmpy2
from gmpy2 import mpz,mpq,mpfr
from gmpy2 import f2q
import numpy as np
import warnings
import ray
import combalg.subset as subset
import math

gmpy2.get_context().precision=5000
#ray.init(log_to_driver=False)

# get_Bernstein_pvalue(0.2,100,2,5,discrete_moments2(100,10,5))
# get_Bernstein_pvalue(0.2,100,2,5,continuous_moments(100,2,5, wList=[1,1,1,1,1]))


def _single_comp(n,k):
    '''
    Source: https://pythonhosted.org/combalg-py/_modules/combalg/combalg.html#composition.random
    Returns a random composition of n into k parts.

    Parameters
    ----------
    n : integer
        integer to compose
    k : integer
        number of parts to compose

    Returns
    -------
    list
        a list of :math:`k` elements which sum to :math:`n`

    '''
    a = subset.random_k_element(range(1, n + k), k - 1)
    r = [0]*k
    r[0] = a[0] - 1
    for j in range(1,k-1):
        r[j] = a[j] - a [j-1] - 1
    r[k-1] = n + k - 1 - a[k-2]
    return r
        

def _compositions(n, k, nsample):
    '''
    Source: https://pythonhosted.org/combalg-py/_modules/combalg/combalg.html#composition.random
    Returns a random composition of :math:`n` into :math:`k` parts.

    Depends on: _single_comp

    Parameters
    ----------
    n : integer
        integer to compose
    k : integer
        number of parts to compose
    nsample: integer
        number of compositions to generate


    Returns
    -------
    2D list
        a list of lists of compositions that sum to n

    '''
    total_compositions = []
    for _ in range(nsample):
        this_comp = []
        for p in _single_comp(n,k):
            this_comp.append(p)
        total_compositions.append(this_comp)
    return np.array(total_compositions)


def get_composition_pvalue(t, n, k, p, wList, alternative, resamp_number, type):
    '''
    Approximate p-value by Resampling Integer Compositions
    
    Given the value of the test statistic :math:`t`, the sample sizes :math:`n` and :math:`k`,
    power exponent :math:`p` and vector of weights that together determine the test statistic
    (by default :math:`n\\geqslant k`), as well as the user-specified resampling number, 
    performs resampling from the collection of integer compositions
    to approximate the p-value of the observed test statistic.

    The function returns a two-sided p-value by default, which is more conservative. Users can
    choose other p-values corresponding to different alternatives; see documentation on `alternative`.
    Note that the interpretation of the choice of `alternative` depends on the choice of weight vector.
    For example, a weight vector that is a quadratic kernel will upweight the extreme components of
    the weight vector. For this choice, setting `alternative` to be `greater` translates into an alternative
    hypothesis of a bigger spread in the larger sample (the one with sample size :math:`n`).

    Depends on: _compositions

    Parameters
    ----------
    t : float
        Value of test statistic :math:`||S_{n,k}(D)/n||_{p,\\boldsymbol{w}}^p` computed from data :math:`D`
    n : integer
        Sample size of :math:`y`
    k : integer
        Sample size of :math:`x`
    p: integer
        Power exponent of test statistic
    wList: list
        Weight vector
    alternative: string
        Character string that should be one of "`two.sided`" (default), "`greater`" or "`less`"
    resamp_number: integer
        Number of compositions of :math:`n` to draw (default is 5000)
    type: string
        Character string that should be one of "`unbiased`", "`valid`" or "`both`"


    Returns
    -------
    float
        p-value
    '''
    # Sample test statistic and compute empirical CDF at t
    resampled_ts = np.matmul(np.power(np.divide(_compositions(n, k, nsample=resamp_number), n), p), wList)
    cdf_at_t = np.mean(resampled_ts < t)
    cdf_at_t_upp_tail = 1 - np.mean(np.append(resampled_ts,[t]) >= t)
    cdf_at_t_low_tail = np.mean(np.append(resampled_ts,[t]) <= t)
    
    if alternative == "two.sided":
        #print("Computing two-sided p-value")
        if type == "unbiased":
            return 2*min(cdf_at_t, 1-cdf_at_t)
        elif type == "valid":
            return 2*min(cdf_at_t_low_tail, 1-cdf_at_t_upp_tail)
        else:
            unbiased = 2*min(cdf_at_t, 1-cdf_at_t)
            valid = 2*min(cdf_at_t_low_tail, 1-cdf_at_t_upp_tail)
            return [unbiased, valid]
            #return "unbiased: " + str(unbiased) + ", valid: " + str(biased)
    
    elif alternative == "greater":
        print("Computing one-sided p-value with alternative set to greater")
        if type == "unbiased":
            return 1-cdf_at_t
        elif type == "valid":
            return 1-cdf_at_t_upp_tail
        else:
            unbiased = 1-cdf_at_t
            valid = 1-cdf_at_t_upp_tail
            return [unbiased, valid]
            #return "unbiased: " + str(unbiased) + ", valid: " + str(biased)
    
    else:
        print("Computing one-sided p-value with alternative set to less")
        if type == "unbiased":
            return cdf_at_t
        elif type == "valid":
            return cdf_at_t_low_tail
        else:
            unbiased = cdf_at_t 
            valid = cdf_at_t_low_tail
            return [unbiased, valid]
            #return "unbiased: " + str(unbiased) + ", valid: " + str(biased)