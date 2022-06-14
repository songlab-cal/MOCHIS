from MOCHIS.auxiliary import *
import numpy as np
import math
import scipy 

def _simplex_sample(n, N): 
    res = []
    for _ in range(N):
        k = np.random.exponential(scale=1.0, size=n) 
        res.append(np.array( k / sum(k)))
    return np.array(res)

def mochis_py(x, p, wList, alternative, approx, n_mom, resamp_number=5000, y=None, force_discrete=False):
    '''
    Flexible Non-Parametric One- and Two-Sample Tests

    Given data consisting of either a single sample \eqn{\boldsymbol{x}=(x_1,\ldots,x_k)}, 
    or two samples \eqn{\boldsymbol{x}=(x_1,\ldots,x_k)} and \eqn{\boldsymbol{y}=(y_1,\ldots,y_n)}, 
    this function uses summary statistics computed on weighted linear combinations of powers of 
    the spacing statistics \eqn{S_k} (former) or \eqn{S_{n,k}} (latter). 

    More precisely, this function does the following:

    For a single sample \eqn{x}, the function tests for uniformity of its entries. When \eqn{p=2}
    and a particular choice of \eqn{\boldsymbol{w}} is specified, we recover Greenwood's test. 

    For two samples, the function tests the null of \eqn{\boldsymbol{x}} and \eqn{\boldsymbol{y}} 
    being drawn from the same distribution (i.e., stochastic equality), against flexible alternatives 
    that correspond to specific choices of the test statistic parameters, \eqn{\boldsymbol{w}} (weight vector) 
    and \eqn{p} (power). These parameters not only determine the test statistic 
    \eqn{||S_k||_{p,\boldsymbol{w}}^p=\sum_{j=1}^k w_iS_{k}[j]^p} (analogously defined for 
    \eqn{||S_{n,k}||_{p,\boldsymbol{w}}^p}), but also encode alternative hypotheses
    ranging from different populational means (i.e., \eqn{\mu_x \neq \mu_y}), different 
    populational spreads (i.e., \eqn{\sigma^2_x \neq \sigma^2_y}), etc. 

    Additional tuning parameters include (1) choice of p-value computation (one- or two-sided);
    (2) approximation method (moment-based such as Bernstein, Chebyshev or Jacobi, or resampling-based); 
    (3) number of moments accompanying the approximation chosen if using moment-based approximation
    (recommended 200, typically at least 100); and (4) in case of two samples,
    whether the user prefers to use exact discrete moments (more accurate but slower) or to use
    continuous approximations of the discrete moments (less accurate but faster).         

    (4/21/22) Currently, only resampling and Gaussian asymptotics are supported. Both are efficient and well-calibrated.

    (4/14/22) Currently, for \eqn{n\geqslant 100} and \eqn{k\geqslant 50} such that \eqn{\frac{k}{n}\geqslant 0.001}, function
    automatically uses Gaussian approximation to the null.
    
    Depends on: functions in `auxiliary.py`

    Parameters
    ----------
    x: Python float list or float numpy array 
        First sample
    y: Python float list or float numpy array 
        Second sample
    p: int
        Exponent value in defining test statistic (must be integer)
    wList: Python float list or float numpy array
        Vector of weights. It should have length equal to \eqn{x} when \eqn{y} is `NULL`,
        and one more than the length of \eqn{x} when \eqn{y} is not `NULL`
        alternative How p-value should be computed; i.e., a character specifying the alternative hypothesis, 
        must be one of "`two.sided`", "`greater`" or "`less`"
    approx: str
        Which approximation method to use (choose `resample`, `bernstein`, `chebyshev`, `jacobi`)
    n_mom: int
        The number of moments to accompany the approximation (recommended 200, if not at least 100)
    resamp_number: int
        Number of \eqn{k}-compositions of \eqn{n} or simplex vectors in \eqn{[0,1]^k}  to draw
        force_discrete In the two-sample case, whether to use discrete moments even if \eqn{n} is large enough (default is `FALSE`)
    
    Returns
    -------
    float
        p-value for non-parametric one- or two-sample tests

    # One-sample examples
    mochis_py(x = [abs(np.random.normal()) for i in range(10)], p = 2, wList = [1 for i in range(10)], alternative = "two.sided", approx = "chebyshev", n_mom = 200) 
    mochis_py(x = [abs(np.random.normal()) for i in range(10)], p = 2, wList = [1 for i in range(10)], alternative = "two.sided", approx = "bernstein", n_mom = 200) 
    mochis_py(x = [abs(np.random.normal()) for i in range(10)], p = 2, wList = [1 for i in range(10)], alternative = "two.sided", approx = "jacobi", n_mom = 100) 
    mochis_py(x = [abs(np.random.normal()) for i in range(10)], p = 2, wList = [1 for i in range(10)], alternative = "two.sided", approx = "resample")

    # Two-sample examples
    mochis_py(x = [abs(np.random.normal()) for i in range(10)], y = [abs(np.random.normal()) for i in range(100)], p = 2, wList = [1 for i in range(11)], alternative = "two.sided", approx = "chebyshev", n_mom = 200) 
    mochis_py(x = [abs(np.random.normal()) for i in range(10)], y = [abs(np.random.normal()) for i in range(100)], p = 2, wList = [1 for i in range(11)], alternative = "two.sided", approx = "bernstein", n_mom = 200) 
    mochis_py(x = [abs(np.random.normal()) for i in range(10)], y = [abs(np.random.normal()) for i in range(100)], p = 2, wList = [1 for i in range(11)], alternative = "two.sided", approx = "jacobi", n_mom = 200) 
    mochis_py(x = [abs(np.random.normal()) for i in range(30)], y = [abs(np.random.normal()) for i in range(100)], p = 2, wList = [1 for i in range(11)], alternative = "two.sided", approx = "resample", resamp_number = 5000)
    '''
    if type != "unbiased" or type != "valid" or type != "both":
        raise IOError("")
    
    # 1. Get numbr of bins
    k = len(x) + 1

    # 2. Normalize weights
    print("Normalizing weight vector...")
    wList = [i/max(wList) for i in wList]

    # 3. Compute test statistic t and return its p-value
    # 3.1 Case 1: y is not NULL
    if y is not None:
        # construct ordering of x_i's
        x_ordered = np.sort(np.array(x))
        x_ordered = np.insert(x_ordered, 0, -math.inf)
        x_ordered = np.append(x_ordered, math.inf) 
        n = len(y) # get sample size / number of balls


        # Construct Snk
        snk = []
        for i in range(k):
            snk.append(((x_ordered[i] <= y) & (y < x_ordered[i+1])).sum())

        
        # Construct t
        t_arr = [((snk[i]/n)**p) * wList[i] for i in range(k)]
        t = sum(t_arr)
        print("The test statistic for the data is ", t)

        # Decide on an approximation:
        # First, decide whether to use large n, large k asymptotics
        if n >= 100 and k >= 50 and k/n >= 1e-3 and (p==1 or p==2):
            print("Sample sizes, n and k, large enough such that k/n > 0; p = 1 or p = 2. Applying Gaussian asymptotics...")
            # [!] 3/27/22 Update: Jonathan and Alan discovered poor FPR using the Monte Carlo method.
            # Changing the implementation to rely on Dan's analytical formulae.
            # Compute analytical mean and variance
            if p==1:
                first_moment = sum(wList) / k 
                second_moment = ((k/n+1)/((k**2)*(k+1))) * (wList *((k*np.identity(k) - np.outer(np.ones(k),np.ones(k))) @ wList)).sum()
                
            else: # p = 2
                first_moment = ((2+k/n-1/n) / ((k+1)*k)) * sum(wList)
                sum_of_wj2s = sum([i**2 for i in wList])
                coeff_sum_of_wj2s = (k-1) * (k/n+1) * (2+k/n-1/n) * (12-6/n+k*(k/n+10-5/n)) / (k^2*(1+k)^2*(2+k)*(3+k))
                offdiag_sum = np.outer(wList, wList).sum() - sum([i**2 for i in wList])
                coeff_offdiag_sum = (k/n+1) * (6/n**2+k*(3+k*(k-2))/n**2-24/n+8*(k-1)*k/n+8*(3+2*k)) / (k**2*(1+k)**2*(2+k)*(3+k))
                second_moment = sum_of_wj2s * coeff_sum_of_wj2s - offdiag_sum * coeff_offdiag_sum

            z_score = (t - first_moment) / second_moment**(1/2)

            if alternative == "one.sided":
                return min(scipy.stats.norm.cdf(z_score), scipy.stats.norm.cdf(-1*z_score))
            else:
                return 2*min(scipy.stats.norm.cdf(z_score), scipy.stats.norm.cdf(-1*z_score))
        elif approx == "resample":
            print("Using resampling approach, with resampling number 5000, to approximate p-value...")
            return get_composition_pvalue(t=t, n=n, k=k, p=p, wList=wList, alternative=alternative, resamp_number=5000, type=unbiased)
        elif n >= 100 and not force_discrete:
            print("Sample size, n, is large enough, using Sk distribution...")

            # compute continuous moments
            print("Computing continuous moments...")
            moment_seq = continuous_moments(m=n_mom, p=p, k=k, wList=wList)

            # compute and return p-value
            if approx == 'bernstein':
                return get_Bernstein_pvalue(t=t, n_mom=n_mom, p=p, k=k, moment_seq=moment_seq, alternative=alternative, wList=wList)
            else:
                return get_moment_pvalue(t=t, n_mom=n_mom, moment_seq=moment_seq, method=approx, alternative=alternative, type="unbiased")
        else:
            print("Using SnK distribution...")
            # compute discrete moments
            print("Computing discrete moments...")

            moment_seq = discrete_moments(m=n_mom, n=n, k=k, p=p, wList=wList)

            # compute and return p-value
            if approx == 'bernstein':
                return get_Bernstein_pvalue(t=t, n_mom=n_mom, p=p, k=k, moment_seq=moment_seq, alternative=alternative, wList=wList)
            else:
                return get_moment_pvalue(t=t, n_mom=n_mom, moment_seq=moment_seq, method=approx, alternative=alternative, type="unbiased")
    
    # y is null -> use continuous moments
    else:
        # check all entries of observations are non-negative
        for entry in x:
            assert entry >= 0
       
        
        # construct Sk
        sum_x = sum(x)
        Sk = [i/sum_x for i in x]

        # construct t
        t = sum((Sk[i]**p)*wList[i] for i in range(len(wList)))

        # decide on approximation
        if approx == "resample":
            print("Using resampling approach on continuous simplex, with resampling number " + str(resamp_number) + ", to approximate p-value...")
            resampled_ts = np.matmul(np.power(_simplex_sample(n=k, N=resamp_number), p), wList)
            
            cdf_at_t = np.mean(resampled_ts < t)
            cdf_at_t_upp_tail = 1 - np.mean(np.append(resampled_ts,[t]) >= t)
            cdf_at_t_low_tail = np.mean(np.append(resampled_ts,[t]) <= t)

            if alternative == "two.sided":
                print("Computing two-sided p-value")
                if type == "unbiased":
                    return 2*min(cdf_at_t, 1-cdf_at_t)
                elif type == "valid":
                    return 2*min(cdf_at_t_low_tail, 1-cdf_at_t_upp_tail)
                else:
                    unbiased = 2*min(cdf_at_t, 1-cdf_at_t)
                    valid = 2*min(cdf_at_t_low_tail, 1-cdf_at_t_upp_tail)
                    return "unbiased: " + str(unbiased) + ", valid: " + str(biased)
            
            elif alternative == "greater":
                print("Computing one-sided p-value with alternative set to greater")
                if type == "unbiased":
                    return 1-cdf_at_t
                elif type == "valid":
                    return 1-cdf_at_t_upp_tail
                else:
                    unbiased = 1-cdf_at_t
                    valid = 1-cdf_at_t_upp_tail
                    return "unbiased: " + str(unbiased) + ", valid: " + str(biased)
            
            else:
                print("Computing one-sided p-value with alternative set to less")
                if type == "unbiased":
                    return cdf_at_t
                elif type == "valid":
                    return cdf_at_t_low_tail
                else:
                    unbiased = cdf_at_t 
                    valid = cdf_at_t_low_tail
                    return "unbiased: " + str(unbiased) + ", valid: " + str(biased)
        
        else:
        
            # construct moments
            moment_seq = continuous_moments(m=n_mom, p=p, k=k, wList=wList)

            # compute and return p-value
            if approx == 'bernstein':
                return get_Bernstein_pvalue(t=t, n_mom=n_mom, p=p, k=k, moment_seq=moment_seq, alternative=alternative, wList=wList)
            else:
                return get_moment_pvalue(t=t, n_mom=n_mom, moment_seq=moment_seq, method=approx, alternative=alternative, type="unbiased")



