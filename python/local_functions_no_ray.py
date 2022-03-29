## This version is modified from test_mpfr_rachel.py

import gmpy2
from gmpy2 import mpz,mpq,mpfr
from gmpy2 import f2q
import numpy as np
import warnings

gmpy2.get_context().precision=3000

def factorial(num):
    '''
    Factorial function for infinite precision integer arithmetic

    Parameters
    ----------
    num : int
        Desired integer whose factorial is to be evaluated with infinite
        precision (i.e., no error beyond :math:`64`-bit float precision)
        
    Raises
    -------
    IOError
        An error is raised if num is negative
    
    Returns
    -------
    mpz
        The resulting factorial
    '''
    if num < 0:
        raise IOError('Cannot take the factorial of a negative number')
    result = mpz(1)
    for i in range(1, num+1):
        result *= mpz(i)
    return result


def choose(top, bottom):
    '''
    Binomial coefficients for infinite precision integer arithmetic

    Parameters
    ----------
    top : int
    bottom : int
    
    Returns
    -------
    mpq
        The resulting binomial coefficient
    '''
    diff = mpz(top) - mpz(bottom)
    numerator = mpq(factorial(top),1)
    denom = mpq(factorial(bottom),1) * mpq(factorial(diff),1)
    
    return numerator/denom


def our_pow(base, exponent):
    '''
    Exponentiation (``mpq``-friendly) 

    Parameters
    ----------
    base : mpq
        The base to which the power function is to be applied
    exponent : integer
        The number of times to raise the base to. 

    Returns
    -------
    res : mpq
        Result of exponentiation

    '''
    res = 1
    while exponent > 0:
        res *= base 
        exponent -= 1
    return res


def iter_pow(base, exponent):
    '''
    Iterative Exponentiation (``mpq``-friendly) 
    
    This function is used to generate a list of successive powers of a base.
    It is more optimal than applying our_pow across a list of powers, since
    any intermediate powers of the base is computed exactly once and stored for
    computing higher powers.

    Parameters
    ----------
    base : mpq
        The base to which the power function is to be applied
    exponent : integer
        The number of times to raise the base to. 

    Returns
    -------
    res : TYPE
        DESCRIPTION.

    '''
    res = [mpq(1,1)]
    while exponent > 0:
        res.append(res[-1]*base)
        exponent -= 1
    return res

# # Test iter_pow vs our_pow
# import time
# import statistics

# iter_pow_benchmarks = []
# our_pow_benchmarks = []

# for iteration in range(20):
#         print("Starting iteration " + str(iteration) + "...")
        
#         iter_pow_time_s = time.time()
#         iter_pow(2,100)
#         iter_pow_time_e = time.time()
#         iter_pow_benchmarks.append(iter_pow_time_e - iter_pow_time_s)
#         our_pow_time_s = time.time()
#         [our_pow(mpq(f2q(2,0),1), i) for i in range(101)]
#         our_pow_time_e = time.time()
#         our_pow_benchmarks.append(our_pow_time_e-our_pow_time_s)

# print("iter_pow statistics: ")
# print(min(iter_pow_benchmarks), statistics.mean(iter_pow_benchmarks), 
#       statistics.median(iter_pow_benchmarks), max(iter_pow_benchmarks))
# print("our_pow statistics: ")
# print(min(our_pow_benchmarks), statistics.mean(our_pow_benchmarks), 
#       statistics.median(our_pow_benchmarks), max(our_pow_benchmarks))
# # End of test

'''
iter_pow statistics: 
2.8848648071289062e-05 4.680156707763672e-05 3.9577484130859375e-05 9.083747863769531e-05
our_pow statistics: 
0.0009768009185791016 0.0017330169677734375 0.0014549493789672852 0.0059871673583984375

iter_pow has 37x gain on average.
'''

def discrete_conv(v, w):
    '''
    Vector Convolution 

    Given two vectors :math:`\\mathbf{v}` and :math:`\\mathbf{w}`,
    with lengths :math:`\\ell` and :math:`m`, generates the 
    length :math:`\\ell` vector :math:`\\mathbf{x}`, whose
    entries are defined by :math:`x_k=\\sum_{i+j=k} v_i w_j`.

    Dependencies: None

    Parameters
    ----------
    v : Python list
        First vector
    w : Python list
        second vector
    
    Raises
    -------
    None
    
    Returns
    -------
    numpy.ndarray of mpq
        The convolution result

    Examples
    -------
    v = [1,2,3]
    w = [4,5,6]
    discrete_conv2(v, w) -> [mpq(4,1) mpq(13,1) mpq(28,1)]

    v = [1,2,3]
    w = [0.1,0.2,0.3,0.4,0.5]
    discrete_conv2(v, w) -> [mpq(1,10) mpq(2,5) mpq(1,1)]
    '''
    L = len(v)
    x = np.convolve(v,w)[:L]
    return x


def list_conv(l1, l2):
    '''
    List Convolution

    Given two lists of vectors :math:`\\mathcal{L}_1` and :math:`\mathcal{L}_2`,
    with sizes :math:`L_1+1` and :math:`L_2+1`, generates the 
    size :math:`L_1+1` list :math:`\\mathcal{L}`, whose
    elements are vectors encoding convolutions of vectors :math:`\\mathcal{L}_1` 
    and :math:`\\mathcal{L}_2`. In particular, if :math:`\\mathcal{L}_1` encodes 
    the coefficients of the bivariate polynomial :math:`P_1(x,y)` of degree 
    :math:`L_1\times M_1` and :math:`\\mathcal{L}_2` encodes those of 
    :math:`P_2(x,y)`, then the result of this operation encodes the coefficients 
    of their product :math:`(P_1P_2)(x,y)` up to the term :math:`x^{L_1}y^{M_1}`. 
    In all applications, we treat :math:`L_1=L_2` and :math:`M_1=M_2`. 

    Dependencies: discrete_conv

    Parameters
    ----------
    l1 : Python 2D list (matrix)
        First matrix
    l2 : Python 2D list (matrix)
        Second matrix
    
    Raises
    -------
    None
    
    Returns
    -------
    2D numpy.ndarray (matrix) of mpq
        The convolution result

    Examples
    -------
    l1 = [[1,3,5], [2,4,6], [0.1,0.2,0.3]] L_1 = M_1 = 3
    l2 = [[7,9,11], [8,10,12], [0.4,0.5,0.6]] L_2 = M_2 = 3
    list_conv2(l1, l2) -> 
        [[mpq(7,1) mpq(30,1) mpq(73,1)]
        [mpq(22,1) mpq(80,1) mpq(182,1)]
        [mpq(171,10) mpq(56,1) mpq(1211,10)]]
    '''
    
    n1 = len(l1) # no. rows for l1
    n2 = len(l2) # no. rows for l2
    
    result_coef_mat = []


    for l in range(n1): # no. rows for output
        per_row_convolved = []
        # for each pair of rows that sums to l, convolve the two rows
        for i in range(n1):
            for j in range(n2):
                if i + j == l:
                    per_row_convolved.append(discrete_conv(l1[i],l2[j]))
        
        result_coef_mat.append(np.sum(np.array(per_row_convolved), axis=0))
    
    return np.array(result_coef_mat)


def fold_conv(l1, big_list):
    '''
    :math:`k`-Fold Convolution

    :math:`k`-fold generalisation of ``listConv``. While ``listConv`` computes 
    coefficients for the product of two bivariate polynomials :math:`P_1(x,y)` 
    and :math:`P_2(x,y)`, ``fold_conv`` computes coefficients for the product 
    of :math:`k` bivariate polynomials, :math:`P_1(x,y),\ldots,P_k(x,y)`. 

    Concretely, given a list of vectors :math:`\\mathcal{L}` (size 
    :math:`L_1`, each vector length :math:`M_1`) and a size :math:`k` collection 
    of list of vectors :math:`\\mathcal{M}` (each list stores vectors), generates 
    a size :math:`k+1` collection of lists of vectors, with each successive list 
    in the output collection corresponding to the output of successively applying 
    ``listConv`` to an input list from :math:`\\mathcal{M}` and an intermediate 
    fold convolution. In particular, List :math:`k+1` of the output collection 
    is the :math:`k`-fold convolution encoding coefficients of :math:`(P_1P_2\cdots P_k)(x,y)`
    up to the term :math:`x^{L_1}y^{M_1}`.  

    Dependencies: list_conv

    Parameters
    ----------
    l1 : Python 2D list (matrix)
        First matrix
    big_list : list of Python 2D lists (list of matrices)
        Collection of lists
    
    Raises
    -------
    None
    
    Returns
    -------
    list of 2D numpy.ndarrays (list of matrices) of mpq
        The convolution result

    Examples
    -------
    l1 = np.array([[1,3,5], [2,4,6], [0.1,0.2,0.3]])
    l2 = np.array([[7,9,11], [8,10,12], [0.4,0.5,0.6]])
    l3 = np.array([[2,2,3], [1,1,2], [0.3,0.3,0.6]])
    big_list = np.array([l2, l3])
    fold_conv(l1, big_list) -> 
        [[[mpq(1,1) mpq(3,1) mpq(5,1)]
        [mpq(2,1) mpq(4,1) mpq(6,1)]
        [mpq(1,10) mpq(1,5) mpq(3,10)]]

        [[mpq(7,1) mpq(30,1) mpq(73,1)]
        [mpq(22,1) mpq(80,1) mpq(182,1)]
        [mpq(171,10) mpq(56,1) mpq(1211,10)]]

        [[mpq(14,1) mpq(74,1) mpq(227,1)]
        [mpq(51,1) mpq(241,1) mpq(707,1)]
        [mpq(583,10) mpq(2593,10) mpq(3733,5)]]]
    '''
    
    full_list = []
    full_list.append(l1)

    k = len(big_list)

    if k == 1:
        full_list.append(list_conv(full_list[k-1], big_list[k-1]))
        return full_list
    else:
        tmp_list = fold_conv(l1, big_list[:k-1])
        tmp_list.append(list_conv(tmp_list[k-1], big_list[k-1]))
        return tmp_list
    


def base_coefficients(m, n, p, w):
    '''
    Generate Base Coefficients

    Generates the matrix of coefficients of the
    bivariate polynomial :math:`P(x,y) = \\sum_{\\ell=0}^m\\sum_{j=0}^n \\frac{j^{p\\ell w^\\ell}}{\\ell!} x^jy^\\ell`, 
    which appears in the computation of moments of :math:`||S_{n,k}||_{p,\\boldsymbol{w}}^p`.

    Given dimensions :math:`m` and :math:`n`, power index :math:`p` and a scalar 
    weight :math:`w`, returns a list containing :math:`m+1` vectors, each vector 
    of length :math:`n+1`.

    Dependencies: None

    Parameters
    ----------
    m: scalar
        moment to be computed
    n: scalar
        sample size / # of balls
    p: scalar
        power index used in test statistic
    w: scalar
        typically an element of user-specified weight vector when choosing test 
        statistic
    
    Returns
    -------
    2D numpy.ndarray (matrix)
        Base coefficients

    Examples
    -------
    base_coefficients2(m=3, n=5, p=1, w=1) ->
        [[mpq(0,1) mpq(1,1) mpq(1,1) mpq(1,1) mpq(1,1) mpq(1,1)]
        [mpq(0,1) mpq(1,1) mpq(2,1) mpq(3,1) mpq(4,1) mpq(5,1)]
        [mpq(0,1) mpq(1,2) mpq(2,1) mpq(9,2) mpq(8,1) mpq(25,2)]
        [mpq(0,1) mpq(1,6) mpq(4,3) mpq(9,2) mpq(32,3) mpq(125,6)]]
    '''

    y = []
    first_row = [mpq(f2q(1,0),1) for i in range(n+1)]
    first_row[0] = mpq(f2q(0,0),1)
    y.append(first_row)
    outerprod = [[0 for j in range(n+1)] for i in range(m)]
    for i in range(m):
        for j in range(n+1):
            first_term = mpq(f2q(w,0),1) ** mpz(i+1) 
            second_term = mpq(f2q(j,0),1) ** mpz(p*mpz(i+1))    #our_pow(mpq(f2q(j,0),1), mpq(f2q(p,0),1)*mpq(f2q(i+1,0),1))
            numerator = first_term * second_term
            outerprod[i][j] = mpq(numerator,factorial(i+1))
            
    y.extend(outerprod)
    return np.array(y)


def base_coefficients_cont(m, p, w):
    '''
    Generate Coefficients for Base Polynomials
    
    The moments of the weighted :math:`p`-norm :math:`||S_k||_{p,\\boldsymbol{w}}^p`
    of a random point :math:`S_k` contained in the continuous :math:`k`-dimensional
    simplex are given by coefficients of certain convolutions of infinite series.
    These coefficients are computed by truncating the series to a sufficiently
    high degree polynomial. This function generates the coefficients of such
    polynomials, which will be convolved at a later step.
    
    Dependencies: None

    Parameters
    ----------
    m : integer
        number of moments to use (typically at least 50)
    p : float
        power index chosen
    w : float
        value of weight

    Returns
    -------
    to_return : Python list of mpq
        The vector containing coefficients of a degree :math:`m` polynomial

    '''
    num = [a*b for a,b in zip([factorial(p*i) for i in range(m+1)], iter_pow(mpq(f2q(w,0),1),m))]
    den = [mpq(1,factorial(i)) for i in range(m+1)]
    to_return = [a*b for a,b in zip(num,den)]
    
    return to_return


def dyn_convolve(vec_list): 
    '''
    Dynamic Convolution
    
    This function takes in a list of lists and returns a vector, where the output
    vector stores elements of the sequential convolution of the input list of 
    vectors. The input list should contain as its elemments lists generated by
    ``base_coefficients_cont``. 
    
    This version leverages dynamic programming to slightly speed up the sequential
    convolution.
    
    Dependencies: discrete_conv

    Parameters
    ----------
    vec_list : list of Python lists 
        A list of lists, all of same length (typically :math:`m+1` where :math:`m`
        is the number of moments)

    Returns
    -------
    numpy.ndarray of mpq
        The result of the iterative convolution

    '''
    if len(vec_list) == 1:
        return vec_list[0]
    else:
        replace_last = discrete_conv(vec_list[-2], vec_list[-1])
        vec_list = vec_list[:-1]
        vec_list[-1] = replace_last
        return dyn_convolve(vec_list)


def post_convolve(m, n, k, p, w_vec):
    '''
    Given desired number of moments :math:`m`, sample size :math:`n`, number of
    bins :math:`k`, power :math:`p` and vector of weights \eqn{\boldsymbol{w}}, 
    which determine the test statistic :math:`||S_{n,k}||_{p,\\boldsymbol{w}}^p`
    entirely, computes a list. The :math:`(n+1)`th entry of the :math:`(m+1)`th 
    vector in the output list (equivalent to :math:`((m+1),(n+1))`th entry of 
    the corresponding matrix) is the un-normalised :math:`m`th moment of 
    :math:`||S_{n,k}||_{p,\\boldsymbol{w}}^p`. It corresponds to eq. (5) of 
    Theorem 1 in Erdmann-Pham et al. (2021+), specifically :math:`[x^ny^m]\\prod_{i=1}^k G(x,w_iy)`.

    Dependencies: base_coefficients, fold_conv

    Parameters
    ----------
    m: scalar
        moment to be computed
    n: scalar
        sample size / # of balls
    k: scalar
        number of bins
    p: scalar
        power index used in test statistic
    w_vec: Python list
        The vector of weights :math:`\boldsymbol{w}` (must be length :math:`k`)

    Returns
    -------
    list of Python lists
        List of vectors storing coefficients of the :math:`k`-fold product of 
        bivariate polynomials (default)

    Examples
    -------
    postConvolve2(m=3, n=5, k=5, p=2, w_vec=[1,1,1,1,1]) ->
        [[mpq(0,1) mpq(0,1) mpq(0,1) mpq(0,1) mpq(0,1) mpq(1,1)]
        [mpq(0,1) mpq(0,1) mpq(0,1) mpq(0,1) mpq(0,1) mpq(5,1)]
        [mpq(0,1) mpq(0,1) mpq(0,1) mpq(0,1) mpq(0,1) mpq(25,2)]
        [mpq(0,1) mpq(0,1) mpq(0,1) mpq(0,1) mpq(0,1) mpq(125,6)]]

    '''
    
    #start = time.time()
    a = base_coefficients(m, n, p, w_vec[0])
    b = []

    #before_base = time.time()
    for j in range(1,k):
        b.append(base_coefficients(m, n, p, w_vec[j]))
    #after_base = time.time()
    #base_time = after_base - before_base
     
    #before_fold = time.time()
    x = fold_conv(a, b)
    #after_fold = time.time()
    #fold_time = after_fold - before_fold

    

    #end = time.time()
    #print(base_time, fold_time, end-start)

    return x[len(b)]


def discrete_moments(m, n, k, p=2.0, wList=None):
    '''
    Given desired number of moments :math:`m`, sample size :math:`n`, number of 
    bins :math:`k`, power :math:`p` and vector of weights :math:`\\boldsymbol{w}`, 
    which determine the test statistic :math:`||S_{n,k}||_{p,\\boldsymbol{w}}^p` 
    entirely, computes a vector. The :math:`i`th element of this list is the 
    :math:`i`th moment of the test statistic. This corresponds to eq. (5) of 
    Theorem 1 of Erdmann-Pham et al. (2021+). 

    This version works with GNU multiple precision arithmetic. 

    Dependencies: post_convolve

    Parameters
    ----------
    m: scalar
        moment to be computed
    n: scalar
        sample size / # of balls
    k: scalar
        number of bins
    p: scalar
       power index used in test statistic
    wList: Python list
        The vector of weights :math:`\\boldsymbol{w}` (must be length :math:`k`)

    Returns
    -------
    list 
        The vector storing moments of the test statistic 

    Examples
    -------
    discrete_moments2(m=3, n=5, k=5) ->
        [1, mpq(1,5), mpq(1,25), mpq(1,125)]
    '''
    my_list = []
    
    for l in range(1,k+1):
        q = [[0 for j in range(n+1)] for i in range(m+1)]
        for i in range(0, m+1):
            for j in range(n+1):
                if j >= l:
                    q[i][j] = factorial(i)/(choose(j-1, l-1)*(j**(p*i)))
        my_list.append(q)

    a = []
    if wList == None:
        wList = [1 for i in range(k)]
    post_mat = post_convolve(m, n, k, p, wList)

    a = np.multiply(np.array(my_list[k-1]), post_mat)
    
    to_return = [row[n] for row in a]
    to_return[0] = 1

    return to_return


def continuous_moments(m, k, wList, p=2.0):
    '''
    Compute Continuous Moments
    
    Given desired number of moments :math:`m`, number of bins :math:`k`, power
    :math:`p` and vector of weights :math:`\\boldsymbol{w}`, which determine
    the test statistic :math:`||S_k||_{p,\\boldsymbol{w}}^p` entirely, computes
    a vector. Each element stores a moment of the test statistic.
    
    Dependencies: dyn_convolve, base_coefficients_cont
    
    Parameters
    ----------
    m : integer
        The number of moments to be computed (typically at least :math:`100`)
    p : float
        power index used in test statistic
    k : integer
        number of bins
    wList : Python list
        list of weights :math:`\\boldsymbol{w}`. 

    Returns
    -------
    to_return : numpy ndarray
        The vector storing coefficients of the :math:`k`-fold product of univariate
        polynomials 

    '''
    # generate F_j(x) for j = 1,...,k
    coeff_list = [base_coefficients_cont(m,p,wj) for wj in wList]
    
    # take convolutions
    to_return = dyn_convolve(coeff_list)
    
    # post-multiply (element-wise) by (k-1)!m!/(pm+k-1)!
    num = np.array([factorial(k-1)*factorial(i) for i in range(m+1)])
    den = np.array([mpq(1,factorial(p*i+k-1)) for i in range(m+1)])
    to_return = num*den*to_return
    
    return to_return.tolist() # must be list for R to recognize as input for get_Bernstein_pvalue


def get_extrema(p, wList):
    '''
    Given parameters wList and p, computes the model-free upper and 
    lower bounds of the test statistic. This allows for more precise
    approximation of CDFs, in case tail behaviour of approximated CDFs
    behave erratically. Note such erratic behaviour can occur for 
    Jacobi approximation.
    
    Dependencies: None

    Parameters
    ----------
    p : scalar
        power index used in test statistic
    wList : Python list
        The vector of weights :math:`\\boldsymbol{w}` (must be length :math:`k`)

    Returns
    -------
    to_return : Python list 
        A list, whose first and second elements are the minimum and maximum
        values

    '''
    to_return = []
    
    # assign minimum value
    den = np.sum((1/((p*np.array(wList))**(1/(p-1)))))**(-p)
    num = 1/(p**(p/(p-1))) * np.sum(1/np.array(wList)**(1/(p-1)))
    to_return.append(num*den)
    
    # assign maximum value
    to_return.append(np.max(np.array(wList)))
    
    return to_return


def get_Bernstein_pvalue(t, n_mom, p, k, moment_seq, alternative="two.sided", wList=None):
    '''
    Bernstein Approximation :math:`p`-value
    
    Given the value of the test statistic, the number of moments used for 
    Bernstein approximation,, the choice of :math:`p`, the list of moments up
    to the desired number of moments, and the choice of alternative 
    (two-sided or one-sided), function computes the approximate :math:`p`-value
    according to Feller's difference formula.
    
    Dependencies: getExtrema

    Parameters
    ----------
    t : float
        value of test statistic 
    n_mom : integer
        number of moments to use (typically at least 100)
    p : float
        power index chosen
    k : integer
        number of bins
    moment_seq : mpq list
        moment sequence computed from either ``discrete_moments`` or ``continuous_moments``
    alternative : string, optional
        The type of alternative to use. The default is two.sided.
    wList : float list, optional
        the vector of weights. The default is None.

    Raises
    ------
    IOError
        Dimensional agreements are checked and failure to satisfy requirements
        triggers errors. Error messages point to the specific error.

    Returns
    -------
    float
        The p-value

    '''
    # assert number of moments is equal to length of moment sequence - 1
    if not len(moment_seq) == n_mom+1:
        raise IOError('Length of moment sequence should be one plus'
                      'no. moments.')
    
    # set wList
    if wList == None: 
        wList = [1 for i in range(k)]
    
    # assert properties of wList
    if not len(wList) == k:
        raise IOError('Weight vector must have length k.')
    
    if not all(np.array(wList) >= 0):
        raise IOError('Weight vector must have non-negative weights.')
    
    if not np.max(np.array(wList)) == 1:
        raise IOError('Maximum weight should be 1.')
    
    # compute difference vector once and cache
    moment_seq_np = np.array(moment_seq)
    summands = []
    diffs = []
    for y in range(1, n_mom+1):
        choose_term = choose(n_mom, y-1)
        
        diff = mpq(f2q(np.diff(moment_seq_np, n=n_mom-y+1)[y-1],0),1)
        diffs.append(diff)
        neg_one = mpq(mpz(-1)**(mpz(n_mom)-mpz(y)+mpz(1)),1) #mpq(our_pow(-1,n_mom-y+1),1)
        summands.append(choose_term*diff*neg_one)
    
    # compute model-free extrema of test statistic
    extrema = get_extrema(p, wList)
    
    if (t<extrema[0] or t>extrema[1]):
        warnings.warn('The test statistic value exceeds model-free bounds'
                      'for the test statistic.')
        return(0)
    else:
        to_include = []
        for y in range(n_mom):
            if t >= y/n_mom:
                to_include.append(1)
            else:
                to_include.append(0)
    
        to_return = np.sum(np.multiply(np.array(summands), np.array(to_include)))
        to_return = float(mpfr(to_return, 53))
        
        if not (to_return <= 1 and to_return >= 0):
            raise IOError('Bernstein approximation is inaccurate.')
        if alternative == 'two.sided':
            return 2 * np.min([to_return, 1-to_return])
        else:
            return np.min([to_return, 1-to_return])

# get_Bernstein_pvalue(0.2,100,2,5,discrete_moments2(100,10,5))
# get_Bernstein_pvalue(0.2,100,2,5,continuous_moments(100,2,5, wList=[1,1,1,1,1]))

def get_moment_pvalue(t, n_mom, moment_seq, method="chebyshev", alternative="two.sided"):
    '''
    Chebyshev and Jacobi Approximation :math:`p`-value
    
    Given the value of the test statistic, the number of moments used for 
    approximation, the list of moments up to the desired number of moments,
    the method of approximation by moments, and the choice of alternative 
    (two-sided or one-sided), function computes the approximate :math:`p`-value 
    according to an inner product approximation.
    
    Dependencies: None

    Parameters
    ----------
    t : float
        value of test statistic
    n_mom : integer
        number of moments to use (typically at least 100)
    moment_seq : mpq list
        moment sequence computed from either ``discrete_moments`` or ``continuous_moments``
    method : string, optional
        either chebyshev or jacobi. The default is chebyshev.
    alternative : string, optional
        The type of alternative to use. The default is two.sided.

    Raises
    ------
    IOError
        Dimensional agreements and appropriate arguments are checked and failure 
        to satisfy requirements triggers errors. Error messages point to the 
        specific error.

    Returns
    -------
    float
        The p-value.

    '''
    if not (method == 'chebyshev' or method == 'jacobi'):
        raise IOError('Please choose either `chebyshev` or `jacobi` as'
                      'approximation method.')
    
    if method == 'chebyshev':
        moment_seq_trunc = moment_seq[:n_mom]
    else:
        moment_seq_trunc = moment_seq[:4*(n_mom//4)+1]
        
    # find closest value in pre-computed basket of interpolating polynomials
    closest_index = np.argmin(np.abs(t-np.linspace(0, 1, 501)))
    dec = np.linspace(0, 1, 501)[closest_index] # convert to character and paste into string
    
    # open relevant interpolating polynomial
    # [!] To do: Change beginning of path directory when packaging
    file_name = method + "/m" + str(n_mom) + "/coeff_list_m" + str(n_mom) + "_" + \
            str(np.round(dec,3)) + ".txt"
    
    lagrange_poly_coefs = open(file_name, 'r').read().split('\n')
    lagrange_poly_coefs = np.array([mpfr(str(x)) for x in lagrange_poly_coefs])
    precise_moments = np.array([mpq(x) for x in moment_seq_trunc])
    
    # compute inner product
    to_return = sum(np.multiply(precise_moments,lagrange_poly_coefs)) # is sum faster than np.sum for ndarrays?
    
    # assert that inner product is bounded in [0,1]
    if not (to_return <= 1 and to_return >= 0):
            raise IOError(method + ' approximation is inaccurate.')
    if alternative == 'two.sided':
        return float(2 * np.min([to_return, 1-to_return]))
    else:
        return float(np.min([to_return, 1-to_return]))

# get_moment_pvalue(0.2,100,discrete_moments2(100,10,5))
# get_moment_pvalue(0.2,100,continuous_moments(100,2,5, wList=[1,1,1,1,1]))