from local_functions import *
import numpy as np
import math
import statistics
import scipy 

######################################################
######## Main Function for Computing p-values ########
######################################################

'''
Flexible Non-Parametric One- and Two-Sample Tests

Given data consisting of either a single sample \eqn{\boldsymbol{x}=(x_1,\ldots,x_k)}, or two samples
'''

def mochis_py(x, p, wList, alternative, approx, n_mom, y=None, force_discrete=False):

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

        # Decides whether to use large n asymptotics or not
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
        elif n >= 100 and not force_discrete:
            print("Sample size, n, is large enough, using Sk distribution...")

            # compute continuous moments
            print("Computing continuous moments...")
            moment_seq = continuous_moments(m=n_mom, p=p, k=k, wList=wList)

            # compute and return p-value
            if approx == 'bernstein':
                return get_Bernstein_pvalue(t=t, n_mom=n_mom, p=p, k=k, moment_seq=moment_seq, alternative=alternative, wList=wList)
            else:
                return get_moment_pvalue(t=t, n_mom=n_mom, moment_seq=moment_seq, method=approx, alternative=alternative)
        else:
            print("Using SnK distribution...")
            # compute discrete moments
            print("Computing discrete moments...")
            moment_seq = discrete_moments(m=n_mom, n=n, k=k, p=p, wList=wList)

            # compute and return p-value
            if approx == 'bernstein':
                return get_Bernstein_pvalue(t=t, n_mom=n_mom, p=p, k=k, moment_seq=moment_seq, alternative=alternative, wList=wList)
            else:
                return get_moment_pvalue(t=t, n_mom=n_mom, moment_seq=moment_seq, method=approx, alternative=alternative)
    
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
        
        # construct moments
        moment_seq = continuous_moments(m=n_mom, p=p, k=k, wList=wList)

        # compute and return p-value
        if approx == 'bernstein':
            return get_Bernstein_pvalue(t=t, n_mom=n_mom, p=p, k=k, moment_seq=moment_seq, alternative=alternative, wList=wList)
        else:
            return get_moment_pvalue(t=t, n_mom=n_mom, moment_seq=moment_seq, method=approx, alternative=alternative)

import random

if __name__ == "__main__":
    x0=[0.9001420, 1.1733458, 0.8974854, 1.4445014, 0.3310136, 2.9006290, 1.0592557, 0.2779547, 0.7494859, 0.2415825]
    y0=[1.00618570, 0.18514603, 0.98182671, 0.09290795, 0.05278440, 0.08032790, 0.65410367, 0.95068351, 1.01956176, 0.85904641, 0.36446078, 0.38365100,
        1.11340572, 1.21150979, 0.34832546, 0.85955345, 0.65002719, 0.32805913, 0.51794657, 0.23898215, 0.11777886, 0.83151845, 1.55891892, 0.22051907,
        0.81719443, 1.07667744, 1.07966400, 0.14212810, 0.15697955, 0.16872026, 0.26903769, 0.80776842, 1.12471724, 1.43078802, 0.06035668, 0.79298250,
        0.34027593, 0.25946873, 1.30484865, 0.36817344, 1.69319002, 0.99583702, 0.18675214, 1.23833740, 0.30937330, 0.63571793, 0.02318327, 1.17786360,
        0.45354660, 0.41592754, 1.33844239, 1.29197474, 0.30907421, 0.15651209, 0.83391675, 0.02454928, 1.13735155, 1.07205425, 2.31449858, 0.42297310,
        0.13693905, 1.32839652, 0.43653567, 0.06642851, 1.30465981, 0.20925954, 1.01825304, 1.36603496, 1.47669585, 0.88730653, 1.01590890, 1.87086906,
        1.07619842, 1.07440763, 2.19557597, 0.53454457, 1.34373343, 1.38500354, 2.74692678, 0.04594469, 0.74308534, 0.26042470, 0.42828190, 0.36825875,
        2.88742330, 0.60707264, 1.87919874, 0.71835657, 0.25142557, 0.46702297, 0.21788838, 1.21764771, 0.75407851, 0.19123018, 1.04728276, 1.08982839,
        0.69547442, 0.52436876, 0.26401593, 1.24120039]
    np.random.seed(2022)
    
    #x0 = [abs(np.random.normal()) for i in range(10)]
    #y0 = [abs(np.random.normal()) for i in range(100)]

    
    #x0 = [abs(np.random.normal()) for i in range(10)]
    #print(x0)
    #y0 = [abs(np.random.normal()) for i in range(100)]
    #print(y0)
    #print(mochis_py(x=x0, y=y0, p=2, wList=[1,1,1,1,1,1,1,1,1,1], alternative="two.sided", approx="chebyshev", n_mom=50))
    #print(mochis_py(x=x0, y=y0, p=2, wList=[1,1,1,1,1,1,1,1,1,1], alternative="two.sided", approx="bernstein", n_mom=200))
    #x1 = [abs(np.random.normal()) for i in range(10)]
    #y1 = [abs(np.random.normal()) for i in range(50)]
    #print(mochis_py(x=x1, y=y1, p=2, wList=[1,1,1,1,1,1,1,1,1,1], alternative="two.sided", approx="bernstein", n_mom=150))
    #print(mochis_py(x=x1, y=y1, p=2, wList=[1,1,1,1,1,1,1,1,1,1], alternative="two.sided", approx="bernstein", n_mom=50))

    # Simulate samples

   

    print(mochis_py(x=x0, y=y0, p=2, wList=[1 for i in range(11)], alternative="two.sided", approx="chebyshev", n_mom=200))
    