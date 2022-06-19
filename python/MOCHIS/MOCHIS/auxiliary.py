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

def _factorial(num):
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


def _choose(top, bottom):
    '''
    Binomial coefficients for infinite precision integer arithmetic

    Parameters
    ----------
    top : int
    bottom : int

    Raises
    -------
    None
    
    Returns
    -------
    mpq
        The resulting binomial coefficient
    '''
    diff = mpz(top) - mpz(bottom)
    numerator = mpq(_factorial(top),1)
    denom = mpq(_factorial(bottom),1) * mpq(_factorial(diff),1)
    
    return numerator/denom


def _our_pow(base, exponent):
    '''
    Exponentiation (``mpq``-friendly) 

    Parameters
    ----------
    base : mpq
        The base to which the power function is to be applied
    exponent : integer
        The number of times to raise the base to. 

    Raises
    -------
    None

    Returns
    -------
    result : mpq
        Result of exponentiation

    '''
    result = 1
    while exponent > 0:
        result *= base 
        exponent -= 1
    return result


def _iter_pow(base, exponent):
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
    exponent : int
        The number of times to raise the base to. 
    
    Raises
    -------
    None

    Returns
    -------
    result : Python mpq list
        A list of incremental power values

    '''
    result = [mpq(1,1)]
    while exponent > 0:
        result.append(result[-1]*base)
        exponent -= 1
    return result


@ray.remote 
def _discrete_conv(v, w):
    '''
    Vector Convolution (with ray)

    Given two vectors :math:`\\mathbf{v}` and :math:`\\mathbf{w}`,
    with lengths :math:`\\ell` and :math:`m`, generates the
    length :math:`\\ell` vector :math:`\\mathbf{x}`, whose
    entries are defined by :math:`x_k=\\sum_{i+j=k} v_i w_j`. This
    version stores the vector as a mpq object.

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
    _discrete_conv(v, w) -> [mpq(4,1) mpq(13,1) mpq(28,1)]

    v = [1,2,3]
    w = [0.1,0.2,0.3,0.4,0.5]
    _discrete_conv(v, w) -> [mpq(1,10) mpq(2,5) mpq(1,1)]
    '''
    L = len(v)
    x = np.convolve(v,w)[:L]
    return x

def _discrete_conv_no_ray(v, w):
    '''
    Vector Convolution (without  ray)

    Vector Convolution (with ray)

    Given two vectors :math:`\\mathbf{v}` and :math:`\\mathbf{w}`,
    with lengths :math:`\\ell` and :math:`m`, generates the
    length :math:`\\ell` vector :math:`\\mathbf{x}`, whose
    entries are defined by :math:`x_k=\\sum_{i+j=k} v_i w_j`. This
    version stores the vector as a mpq object.

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
    _discrete_conv_no_ray(v, w) -> [mpq(4,1) mpq(13,1) mpq(28,1)]

    v = [1,2,3]
    w = [0.1,0.2,0.3,0.4,0.5]
    _discrete_conv_no_ray(v, w) -> [mpq(1,10) mpq(2,5) mpq(1,1)]
    '''
    L = len(v)
    x = np.convolve(v,w)[:L]
    return x


def _list_conv(l1, l2):
    '''
    List Convolution

    Given two lists of vectors :math:`\\mathcal{L}_1` and :math:`\\mathcal{L}_2`,
    with sizes :math:`L_1+1` and :math:`L_2+1`, generates the
    size :math:`L_1+1` list :math:`\\mathcal{L}`, whose
    elements are vectors encoding convolutions of vectors :math:`\\mathcal{L}_1` and :math:`\\mathcal{L}_2`.
    In particular, if :math:`\\mathcal{L}_1` encodes the coefficients of the
    bivariate polynomial :math:`P_1(x,y)` of degree :math:`L_1\\times M_1` and :math:`\\mathcal{L}_2`
    encodes those of :math:`P_2(x,y)`, then the result of this operation encodes the
    coefficients of their product :math:`(P_1P_2)(x,y)` up to the term :math:`x^{L_1}y^{M_1}`. In all
    applications, we treat :math:`L_1=L_2` and :math:`M_1=M_2`.

    Dependencies: _discrete_conv

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
    _list_conv2(l1, l2) -> 
        [[mpq(7,1) mpq(30,1) mpq(73,1)]
        [mpq(22,1) mpq(80,1) mpq(182,1)]
        [mpq(171,10) mpq(56,1) mpq(1211,10)]]
    '''
    
    L1 = len(l1) # no. rows for l1
    L2 = len(l2) # no. rows for l2

    
    conv_result_matrix = []

    intermediate_ray_results = [[] for i in range(L1)] # used for speeding up ray operations

    for l in range(L1): 
        for i in range(L1):
            for j in range(L2):
                if i + j == l:
                    
                    intermediate_ray_results[l].append(_discrete_conv.remote(l1[i],l2[j]))


    for row in intermediate_ray_results:
        conv_result_matrix.append(np.sum(np.array(ray.get(row)), axis=0))

    return np.array(conv_result_matrix)



def _fold_conv(l, list_collection):

    '''
    :math:`k`-Fold Convolution

    :math:`k`-fold generalisation of `listConv`. While `listConv` computes coefficients
    for the product of two bivariate polynomials :math:`P_1(x,y)` and :math:`P_2(x,y)`,
    `foldConv` computes coefficients for the product of :math:`k` bivariate polynomials,
    :math:`P_1(x,y),\\ldots,P_k(x,y)`.

    Concretely, given a list of vectors :math:`\\mathcal{L}` (size :math:`L_1`, each vector length :math:`M_1`)
    and a size :math:`k` collection of list of vectors :math:`\\mathcal{M}` (each list stores vectors),
    generates a size :math:`k+1` collection of lists of vectors, with each successive list in the output
    collection corresponding to the output of successively applying `listConv` to an input list from
    :math:`\\mathcal{M}` and an intermediate fold convolution. In particular, List :math:`k+1` of the
    output collection is the :math:`k`-fold convolution encoding coefficients of :math:`(P_1P_2\\cdots P_k)(x,y)`
    up to the term :math:`x^{L_1}y^{M_1}`s.

    This function outputs all vectors in lists as mpq objects.

    Dependencies: _list_conv

    Parameters
    ----------
    l : Python 2D list (matrix)
        First matrix
    list_collection : list of Python 2D lists (list of matrices)
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
    l = np.array([[1,3,5], [2,4,6], [0.1,0.2,0.3]])
    l2 = np.array([[7,9,11], [8,10,12], [0.4,0.5,0.6]])
    l3 = np.array([[2,2,3], [1,1,2], [0.3,0.3,0.6]])
    list_collection = np.array([l2, l3])
    _fold_conv(l, list_collection) -> 
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
    
    conv_result_matrix = []
    conv_result_matrix.append(l)

    k = len(list_collection)

    if k == 1:
        conv_result_matrix.append(_list_conv(conv_result_matrix[k-1], list_collection[k-1]))
        return conv_result_matrix
    else:
        partial_fold_conv = _fold_conv(l, list_collection[:k-1])
        partial_fold_conv.append(_list_conv(partial_fold_conv[k-1], list_collection[k-1]))
        return partial_fold_conv
    


def _base_coefficients(m, n, p, w):
    '''
    Generate Base Coefficients

    Generates the matrix of coefficients of the
    bivariate polynomial :math:`P(x,y) = \\sum_{\\ell=0}^m\\sum_{j=0}^n \\frac{j^{p\\ell w^\\ell}}{\\ell!} x^jy^\\ell`,
    which appears in the computation of moments of :math:`||S_{n,k}||_{p,\\boldsymbol{w}}^p`.

    Given dimensions :math:`m` and :math:`n`, power index :math:`p` and a scalar weight :math:`w`,
    returns a list containing :math:`m+1` vectors, each vector of length :math:`n+1`.

    This function works with GNU multiple precision arithmetic.
    of length :math:`n+1`.

    Dependencies: None

    Parameters
    ----------
    m: int
        moment to be computed
    n: int
        sample size / # of balls
    p: float
        power index used in test statistic
    w: float
        typically an element of user-specified weight vector when choosing test 
        statistic
    
    Raises
    -------
    None
    
    Returns
    -------
    2D numpy.ndarray (matrix)
        Base coefficients

    Examples
    -------
    _base_coefficients2(m=3, n=5, p=1, w=1) ->
        [[mpq(0,1) mpq(1,1) mpq(1,1) mpq(1,1) mpq(1,1) mpq(1,1)]
        [mpq(0,1) mpq(1,1) mpq(2,1) mpq(3,1) mpq(4,1) mpq(5,1)]
        [mpq(0,1) mpq(1,2) mpq(2,1) mpq(9,2) mpq(8,1) mpq(25,2)]
        [mpq(0,1) mpq(1,6) mpq(4,3) mpq(9,2) mpq(32,3) mpq(125,6)]]
    '''

    base_coeffs = []
    first_row = [mpq(f2q(1,0),1) for i in range(n+1)]
    first_row[0] = mpq(f2q(0,0),1)
    base_coeffs.append(first_row)
    
    outerprod = [[0 for j in range(n+1)] for i in range(m)]
    for i in range(m):
        for j in range(n+1):
            first_term = mpq(f2q(w,0),1) ** mpz(i+1) 
            second_term = mpq(f2q(j,0),1) ** mpz(p*mpz(i+1))    #our_pow(mpq(f2q(j,0),1), mpq(f2q(p,0),1)*mpq(f2q(i+1,0),1))
            numerator = first_term * second_term
            outerprod[i][j] = mpq(numerator,_factorial(i+1))
            
    base_coeffs.extend(outerprod)
    return np.array(base_coeffs)


def _base_coefficients_cont(m, p, w):
    '''
    Generate Coefficients for Base Polynomials
    
    The moments of the weighted :math:`p`-norm :math:`||S_k||_{p,\\boldsymbol{w}}^p` of a
    random point :math:`S_k` contained in the continuous :math:`k`-dimensional simplex
    are given by coefficients of certain convolutions of infinite series. These
    coefficients are computed by truncating the series to a sufficiently high degree
    polynomial. This function generates the coefficients of such polynomials, which
    will be convolved at a later step.
    
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
    base_coeffs : Python list of mpq
        The vector containing coefficients of a degree :math:`m` polynomial

    Examples
    -------
    _base_coefficients_cont(m=10, p=2, w=1)

    '''
    num = [a*b for a,b in zip([_factorial(p*i) for i in range(m+1)], _iter_pow(mpq(f2q(w,0),1),m))]
    den = [mpq(1,_factorial(i)) for i in range(m+1)]
    base_coeffs = [a*b for a,b in zip(num,den)]
    
    return base_coeffs


def _dyn_convolve(vec_list): 
    '''
    Dynamic Convolution
    
    This function takes in a list of vectors and returns a vector, where
    the output vector stores elements of the sequential convolution of the
    input list of vectors. The input list should contain as its elements
    vectors generated by `_base_coefficients_cont`.
    
    This version leverages dynamic programming to slightly speed up the sequential
    convolution.
    
    Dependencies: _discrete_conv

    Parameters
    ----------
    vec_list : list of Python lists 
        A list of vectors, all of same length (typically :math:`m+1` where :math:`m` is number of moments)
    
    Examples
    ----------
    k = 5
    coeff_list = [_base_coefficients_cont(10, 2, 1/j) for j in range(1, k+1)]
    _dyn_convolve(coeff_list)

    Returns
    -------
    numpy.ndarray of mpq
        A vector with same length as each vector element in input list (also length :math:`m+1` typically)

    '''
    if len(vec_list) == 1:
        return vec_list[0]
    else:
        replace_last = _discrete_conv_no_ray(vec_list[-2], vec_list[-1])
        vec_list = vec_list[:-1]
        vec_list[-1] = replace_last
        return _dyn_convolve(vec_list)


def _post_convolve(m, n, k, p, w_vec):
    '''
    Given desired number of moments :math:`m`, sample size :math:`n`, number of bins :math:`k`, power :math:`p` and
    vector of weights :math:`\\boldsymbol{w}`, which determine the test statistic :math:`||S_{n,k}||_{p,\\boldsymbol{w}}^p`
    entirely, computes a list. The :math:`(n+1)`th entry of the :math:`(m+1)`th vector in the output list
    (equivalent to :math:`((m+1),(n+1))`th entry of the corresponding matrix) is the un-normalised :math:`m`th moment
    of :math:`||S_{n,k}||_{p,\\boldsymbol{w}}^p`. It corresponds to eq. (5) of Theorem 1 in Erdmann-Pham et al. (2021+),
    specifically :math:`[x^ny^m]\\prod_{i=1}^k G(x,w_iy)`.

    This function works with GMPY multiple precision arithmetic.

    Dependencies: _base_coefficients, _fold_conv

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
        The vector of weights :math:`\\boldsymbol{w}` (must be length :math:`k`)

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
    
    base_coeffs_first_weight = _base_coefficients(m, n, p, w_vec[0])
    base_coeffs_remaining_weights = []

    for j in range(1,k):
        base_coeffs_remaining_weights.append(_base_coefficients(m, n, p, w_vec[j]))
    
    poly_coeffs = _fold_conv(base_coeffs_first_weight, base_coeffs_remaining_weights)
    
    return poly_coeffs[len(base_coeffs_remaining_weights)]


def discrete_moments(m, n, k, p=2, wList=None):
    '''
    Given desired number of moments :math:`m`, sample size :math:`n`, number of bins :math:`k`,
    power :math:`p` and vector of weights :math:`\\boldsymbol{w}`, which determine the test
    statistic :math:`||S_{n,k}||_{p,\\boldsymbol{w}}^p` entirely, computes a vector.
    The :math:`i`th element of this vector is the :math:`i`th moment of the test statistic.
    This corresponds to eq. (5) of Theorem 1 in Erdmann-Pham et al. (2021+).

    This version works with GMPY multiple precision arithmetic. 

    Dependencies: _post_convolve

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
        The vector of weights :math:`\\boldsymbol{w}` (must be length :math:`k`, default is `rep(1,k)`)

    Returns
    -------
    list 
        The vector storing moments of the test statistic 

    Examples
    -------
    discrete_moments2(m=3, n=5, k=5) ->
        [1, mpq(1,5), mpq(1,25), mpq(1,125)]
    '''
    choose_terms = []
    for l in range(1,k+1):
        q = [[0 for j in range(n+1)] for i in range(m+1)]
        for i in range(0, m+1):
            for j in range(n+1):
                if j >= l:
                    q[i][j] = _factorial(i)/(_choose(j-1, l-1)*(j**(p*i)))
        choose_terms.append(q)

    if wList == None:
        wList = [1 for i in range(k)]

    post_mat = _post_convolve(m, n, k, p, wList)

    moments = np.multiply(np.array(choose_terms[k-1]), post_mat)
    
    selected_moments = [row[n] for row in moments]
    selected_moments[0] = 1
    return selected_moments


def continuous_moments(m, k, wList=None, p=2):
    '''
    Compute Continuous Moments
    
    Given desired number of moments :math:`m`, number of bins :math:`k`, power
    :math:`p` and vector of weights :math:`\\boldsymbol{w}`, which determine
    the test statistic :math:`||S_k||_{p,\\boldsymbol{w}}^p` entirely, computes
    a vector. Each element stores a moment of the test statistic.
    
    Dependencies: _dyn_convolve, _base_coefficients_cont
    
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
    moments : numpy ndarray
        The vector storing coefficients of the :math:`k`-fold product of univariate
        polynomials 

    '''
    if wList == None:
        wList = [1 for i in range(k)]
    # generate F_j(x) for j = 1,...,k
    base_coeffs = [_base_coefficients_cont(m,p,wj) for wj in wList]
    
    # take convolutions
    convolutions = _dyn_convolve(base_coeffs)
    
    # post-multiply (element-wise) by (k-1)!m!/(pm+k-1)!
    num = np.array([_factorial(k-1)*_factorial(i) for i in range(m+1)])
    den = np.array([mpq(1,_factorial(p*i+k-1)) for i in range(m+1)])
    moments = num*den*convolutions
    
    return moments.tolist() # must be list for R to recognize as input for get_Bernstein_pvalue

## HERE ##
def _get_extrema(p, wList):
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
    bounds : Python list 
        A list, whose first and second elements are the minimum and maximum
        values

    '''
    bounds = []
    
    # assign minimum value
    den = np.sum((1/((p*np.array(wList))**(1/(p-1)))))**(-p)
    num = 1/(p**(p/(p-1))) * np.sum(1/np.array(wList)**(1/(p-1)))
    bounds.append(num*den)
    
    # assign maximum value
    bounds.append(np.max(np.array(wList)))
    
    return bounds


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
        choose_term = _choose(n_mom, y-1)
        
        diff = mpq(f2q(np.diff(moment_seq_np, n=n_mom-y+1)[y-1],0),1)
        diffs.append(diff)
        neg_one = mpq(mpz(-1)**(mpz(n_mom)-mpz(y)+mpz(1)),1) #mpq(our_pow(-1,n_mom-y+1),1)
        summands.append(choose_term*diff*neg_one)
    
    # compute model-free extrema of test statistic
    extrema = _get_extrema(p, wList)
    
    if (t<extrema[0] or t>extrema[1]):
        warnings.warn('The test statistic value exceeds model-free bounds'
                      'for the test statistic.')
        print("Theoretical maximum = ", extrema[1], "; theoretical minimum = ", extrema[0], "; observed statistic = ", t)
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
        
        #if not (to_return <= 1 and to_return >= 0):
            #raise IOError('Bernstein approximation is inaccurate.')
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
    seq = np.linspace(0, 0.998, 500)
    closest_index = np.argmin([np.abs(t-i) for i in seq])

    dec = seq[closest_index] # convert to character and paste into string

    
    # open relevant interpolating polynomial
    # [!] To do: Change beginning of path directory when packaging
    file_name = method + "/m" + str(n_mom) + "/coeff_list_m" + str(n_mom) + "_" + \
            str(dec) + ".txt"

    lagrange_poly_coefs = open(file_name, 'r').read().split('\n')
    lagrange_poly_coefs = np.array([mpfr(str(x)) for x in lagrange_poly_coefs])
    precise_moments = np.array([mpq(x) for x in moment_seq_trunc])

    
    # compute inner product
    to_return = sum(np.multiply(precise_moments,lagrange_poly_coefs)) # is sum faster than np.sum for ndarrays?
    
    # assert that inner product is bounded in [0,1]
    #if not (to_return <= 1 and to_return >= 0):
            #raise IOError(method + ' approximation is inaccurate.')
    if alternative == 'two.sided':
        return float(2 * np.min([to_return, 1-to_return]))
    else:
        return float(np.min([to_return, 1-to_return]))



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