########################################################
#### This is a backend R script for main functions. ####
### Incorporates all methods (discrete, continuous)  ###
############## Created on Jan 27, 2022 #################
############### Edited on Mar 24, 2022 #################
############### Edited on Apr 15, 2022 #################
########################################################

# Try gmp rational and integer approach
# R double is 64-bit precision with 52 bits used to store decimals

library(Rmpfr)
library(gmp)
library(doParallel)
library(hitandrun) # for simplex simulations
library(arrangements) # for random compositions

#' Vector Convolution
#'
#' Given two vectors \eqn{\mathbf{v}} and \eqn{\mathbf{w}},
#' with lengths \eqn{\ell} and \eqn{m}, generates the 
#' length \eqn{\ell} vector \eqn{\mathbf{x}}, whose
#' entries are defined by \eqn{x_k=\sum_{i+j=k} v_i w_j}. This
#' version stores the vector as a bigq object.
#'
#' This function works with GNU multiple precision arithmetic.
#'  
#' Dependencies: None
#' 
#' @param v First vector 
#' @param w Second vector 
#' @return Their convolution (bigq object)
#' @examples
#' v <- c(1,2,3)
#' w <- c(4,5,6)
#' discreteConv(v,w) 
#' # Big Rational ('bigq') object of length 3:
#' # [1] 4  13 28
#' 
discreteConv <- function(v,w){
  L <- length(v)
  x <- c_bigq(lapply(1:L, function(i) sum(as.vector.bigq(v)[1:i] * rev(as.vector.bigq(w)[1:i]))))
  return(x)
}

#' List Convolution
#'
#' Given two lists of vectors \eqn{\mathcal{L}_1} and \eqn{\mathcal{L}_2},
#' with sizes \eqn{L_1+1} and \eqn{L_2+1}, generates the 
#' size \eqn{L_1+1} list \eqn{\mathcal{L}}, whose
#' elements are vectors encoding convolutions of vectors \eqn{\mathcal{L}_1} and \eqn{\mathcal{L}_2}.
#' In particular, if \eqn{\mathcal{L}_1} encodes the coefficients of the 
#' bivariate polynomial \eqn{P_1(x,y)} of degree \eqn{L_1\times M_1} and \eqn{\mathcal{L}_2}
#' encodes those of \eqn{P_2(x,y)}, then the result of this operation encodes the 
#' coefficients of their product \eqn{(P_1P_2)(x,y)} up to the term \eqn{x^{L_1}y^{M_1}}. In all
#' applications, we treat \eqn{L_1=L_2} and \eqn{M_1=M_2}.     
#'
#' This function stores each vector in the output list as a bigq object. It also efficiently
#' convolves each pair of element vectors of the two input lists, using `foreach`. 
#' 
#' Dependencies: discreteConv
#' 
#' @param l1 First list
#' @param l2 Second list
#' @return List of vectors encoding convolutions 
#' @examples
#' l1 <- list(a = c(1,3,5), b = c(2,4,6), c = c(0.1,0.2,0.3)) # L_1 = M_1 = 3
#' l2 <- list(a = c(7,9,11), b = c(8,10,12), c = c(0.4,0.5,0.6)) # L_2 = M_2 = 3
#' listConv(l1,l2) 
#' # first vector is c(7,30,73), second is c(22,80,182), third is c(17.1,56.0,121.1), all bigq objects
#' # compare manually multiplying the two bivariate polynomials of degree (3,3) in (x,y)
#' 
listConv <- function(l1, l2) {
  L <- length(l1) # this will be m+1 in practice, where m = no. moments
  # parallelize
  v <- foreach(i=1:L) %dopar% {
    l1_temp <- l1[1:i]
    l2_temp <- rev(l2[1:i])
    res <- lapply(1:i,function(j) discreteConv(l1_temp[[j]], l2_temp[[j]]))
    Reduce("+",res)
  }
  return(v)
}

#' \eqn{k}-Fold Convolution
#'
#' \eqn{k}-fold generalisation of `listConv`. While `listConv` computes coefficients
#' for the product of two bivariate polynomials \eqn{P_1(x,y)} and \eqn{P_2(x,y)},
#' `foldConv` computes coefficients for the product of \eqn{k} bivariate polynomials,
#' \eqn{P_1(x,y),\ldots,P_k(x,y)}. 
#' 
#' Concretely, given a list of vectors \eqn{\mathcal{L}} (size \eqn{L_1}, each vector length \eqn{M_1}) 
#' and a size \eqn{k} collection of list of vectors \eqn{\mathcal{M}} (each list stores vectors), 
#' generates a size \eqn{k+1} collection of lists of vectors, with each successive list in the output 
#' collection corresponding to the output of successively applying `listConv` to an input list from
#' \eqn{\mathcal{M}} and an intermediate fold convolution. In particular, List \eqn{k+1} of the 
#' output collection is the \eqn{k}-fold convolution encoding coefficients of \eqn{(P_1P_2\cdots P_k)(x,y)}
#' up to the term \eqn{x^{L_1}y^{M_1}}.       
#'
#' This function outputs all vectors in lists as bigq objects.
#'  
#' Dependencies: listConv
#' 
#' @param l1 First list
#' @param big_list Collection of lists
#' @return Collection of lists
#' @examples
#' l1 <- list(a = c(1,3,5), b = c(2,4,6), c = c(0.1,0.2,0.3))
#' l2 <- list(a = c(7,9,11), b = c(8,10,12), c = c(0.4,0.5,0.6))
#' l3 <- list(a = c(2,2,3), b = c(1,1,2), c = c(0.3,0.3,0.6))
#' big_list <- list(A = l2, B = l3)
#' foldConv(l1, big_list) 
#' 
foldConv <- function(l1, big_list){
  full_list <- list()
  full_list[[1]] <- l1
  
  for(i in 1:length(big_list)){
    # try parallelized for loop within listconv
    full_list[[i+1]] <- listConv(full_list[[i]], big_list[[i]])
  }
  return(full_list)
}

#' Generate Base Coefficients
#'
#' Generates the matrix of coefficients of the
#' bivariate polynomial \eqn{P(x,y) = \sum_{\ell=0}^m\sum_{j=0}^n \frac{j^{p\ell w^\ell}}{\ell!} x^jy^\ell}, 
#' which appears in the computation of moments of \eqn{||S_{n,k}||_{p,\boldsymbol{w}}^p}.
#' 
#' Given dimensions \eqn{m} and \eqn{n}, power index \eqn{p} and a scalar weight \eqn{w}, 
#' returns a list containing \eqn{m+1} vectors, each vector of length \eqn{n+1}.
#'
#' This function works with GNU multiple precision arithmetic.
#' 
#' Dependencies: None
#' 
#' @param m The moment to be computed (scalar)
#' @param n Sample size / number of balls (scalar)
#' @param p Power index used in test statistic (scalar)
#' @param w Weight, typically an element of user-specified weight vector when choosing test statistic (scalar)
#' @return List of vectors, each vector has length \eqn{n+1} and list size is \eqn{m+1}
#' @examples
#' baseCoefficients(m=3, n=5, p=1, w=1)
#' 
baseCoefficients <- function(m, n, p, w) {
  # Generate matrix of coefficients j^{pl}w^l/l!
  my_bigq_mat <- matrix.bigq(nrow = m + 1, ncol = n + 1)
  my_bigq_mat[1,] <- as.bigq(c(0, rep(1, n)))
  my_bigq_mat[2:(m + 1),] <- gmp::outer(1:m, 0:n, 
                                        function(i,j) 
                                          div.bigq(mul.bigq(pow.bigq(w,i), as.bigq(pow.bigq(j, mul.bigq(p,i)))), 
                                                   as.bigq(gmp::factorialZ(i))))

  # Convert to list
  to_return <- lapply(seq_len(nrow(my_bigq_mat)), function(i) my_bigq_mat[i,])
  
  # Return
  return(to_return)
}

#' Generate Coefficients for Base Polynomials
#'
#' The moments of the weighted \eqn{p}-norm \eqn{||S_k||_{p,\boldsymbol{w}}^p} of a
#' random point \eqn{S_k} contained in the continuous \eqn{k}-dimensional simplex 
#' are given by coefficients of certain convolutions of infinite series. These 
#' coefficients are computed by truncating the series to a sufficiently high degree
#' polynomial. This function generates the coefficients of such polynomials, which
#' will be convolved at a later step.
#' 
#' This version relies on gmp for multiple precision arithmetic.
#'  
#' Dependencies: none
#' @param m Number of moments to use (typically \eqn{\geqslant 50})
#' @param p Power index chosen
#' @param w Value of scalar weight 
#' @return A vector containing coefficients of a degree \eqn{m} polynomial
#' @examples
#' 
#' baseCoefficientsCont(m=10,p=2,w=1)
#' 
baseCoefficientsCont <- function(m, p, w) {
  # Generate vector of coefficients w^l*(pl)!/l!
  to_return <- div.bigq(mul.bigq(as.bigq(w)^(0:m), gmp::factorialZ(p*(0:m))), gmp::factorialZ(0:m))
  
  # Return
  return(to_return)
}

#' Dynamic Convolution
#'
#' This function takes in a list of vectors and returns a vector, where
#' the output vector stores elements of the sequential convolution of the 
#' input list of vectors. The input list should contain as its elements 
#' vectors generated by `baseCoefficientsCont`.   
#' 
#' This version leverages dynamic programming to slightly speed up the 
#' sequential convolution.
#'  
#' Dependencies: discreteConv
#' @param list_of_vecs A list of vectors, all of same length (typically \eqn{m+1} where \eqn{m} is number of moments)
#' @return A vector with same length as each vector element in input list (also length \eqn{m+1} typically) 
#' @examples
#' 
#' k <- 5
#' coeff_list <- lapply(1:k, function(j) {baseCoefficientsCont(10,2,1/j)})
#' dynConvolve(coeff_list)
#' 
dynConvolve <- function(list_of_vecs) {
  if (length(list_of_vecs) == 1) {
    return(list_of_vecs[[1]])
  } else {
    L <- length(list_of_vecs)
    replace_last <- discreteConv(list_of_vecs[[L-1]], list_of_vecs[[L]])
    list_of_vecs <- list_of_vecs[-1]
    list_of_vecs[[L-1]] <- replace_last
    return(dynConvolve(list_of_vecs))
  }
}

#' Compute Unnormalised Moments
#'
#' Given desired number of moments \eqn{m}, sample size \eqn{n}, number of bins \eqn{k}, power \eqn{p} and
#' vector of weights \eqn{\boldsymbol{w}}, which determine the test statistic \eqn{||S_{n,k}||_{p,\boldsymbol{w}}^p}
#' entirely, computes a list. The \eqn{(n+1)}th entry of the \eqn{(m+1)}th vector in the output list 
#' (equivalent to \eqn{((m+1),(n+1))}th entry of the corresponding matrix) is the un-normalised \eqn{m}th moment 
#' of \eqn{||S_{n,k}||_{p,\boldsymbol{w}}^p}. It corresponds to eq. (5) of Theorem 1 in Erdmann-Pham et al. (2021+), 
#' specifically \eqn{[x^ny^m]\prod_{i=1}^k G(x,w_iy)}.  
#' 
#' This function works with GNU multiple precision arithmetic.
#' 
#' Dependencies: baseCoefficients, foldConv
#' 
#' @param m The moment to be computed (scalar)
#' @param n Sample size / number of balls (scalar)
#' @param k Number of bins (scalar)
#' @param p Power index used in test statistic (scalar)
#' @param w_vec The vector of weights \eqn{\boldsymbol{w}} (must be length \eqn{k})
#' @return List of vectors storing coefficients of the \eqn{k}-fold product of bivariate polynomials
#' @examples
#' postConvolve(m=3, n=5, k=5, p=2, w_vec=rep(1,k))
#' 
#' # Extract (m+1,n+1) element
#' postConvolve(m=3, n=5, k=5, p=2, w_vec=rep(1,k))[[4]][6,] # should get 20.83333
#' 
postConvolve <- function(m, n, k, p, w_vec){
  a_ = baseCoefficients(m, n, p, w_vec[1])
  b_ = lapply(1:(k-1), function(j) baseCoefficients(m, n, p, w_vec[j + 1]))
  x_ = foldConv(a_,b_)[[length(b_)+1]]
  return(x_)
}

#' Compute Normalised Moments
#'
#' Given desired number of moments \eqn{m}, sample size \eqn{n}, number of bins \eqn{k}, 
#' power \eqn{p} and vector of weights \eqn{\boldsymbol{w}}, which determine the test 
#' statistic \eqn{||S_{n,k}||_{p,\boldsymbol{w}}^p} entirely, computes a vector. 
#' The \eqn{i}th element of this vector is the \eqn{i}th moment of the test statistic.
#' This corresponds to eq. (5) of Theorem 1 in Erdmann-Pham et al. (2021+).      
#' 
#' This function works with GNU multiple precision arithmetic.
#' 
#' Dependencies: postConvolve
#' 
#' @param m The moment to be computed (scalar)
#' @param n Sample size / number of balls (scalar)
#' @param k Number of bins (scalar)
#' @param p Power index used in test statistic (scalar, default is 2)
#' @param wList The vector of weights \eqn{\boldsymbol{w}} (must be length \eqn{k}, default is `rep(1,k)`)
#' @return Vector storing moments of the test statistic  
#' @examples
#' discreteMoments(m=3, n=5, k=5, p=2)
#' 
discreteMoments <- function(m, n, k, p = 2, wList = rep(1,k)){
  
  my_list = list()
  for(l in 1:k){
    q = gmp::matrix.bigq(0,m+1,n+1)
    for(i in 0:m){
      for(j in 0:n){
        if(j>=l){
          q[i+1,j+1] = div.bigq(factorialZ(i), mul.bigq(chooseZ(j-1, l-1), pow.bigq(j, mul.bigq(p,i))))
        }
      }
    }
    my_list[[l]] = q
  }
  
  post_mat = Reduce(cbind, postConvolve(m,n,k,p,wList))
  a = t(my_list[[k]]) * post_mat
  to_return <- as.vector(a[n+1,]) 
  
  return(to_return) 
}

#' Compute Continuous Moments 
#'
#' Given desired number of moments \eqn{m}, number of bins \eqn{k}, power \eqn{p} and
#' vector of weights \eqn{\boldsymbol{w}}, which determine the test statistic \eqn{||S_k||_{p,\boldsymbol{w}}^p}
#' entirely, computes a vector. Each vector stores moments of the test statistic.
#' 
#' This function works with GNU multiple precision arithmetic.
#' 
#' Dependencies: baseCoefficientsCont, dynConvolve
#' 
#' @param m The number of moments to be computed (scalar)
#' @param k Number of bins (scalar)
#' @param p Power index used in test statistic (scalar)
#' @param wList The vector of weights \eqn{\boldsymbol{w}} (must be length \eqn{k})
#' @return Vector storing coefficients of the \eqn{k}-fold product of univariate polynomials  
#' @examples
#' continuousMoments(m=10,p=2,k=5)
#' 
continuousMoments <- function(m, p, k, wList = rep(1,k)) {
  # Generate F_j(x) for j = 1,...,k
  coeff_list <- lapply(1:k, function(j) {baseCoefficientsCont(m,p,wList[j])})
  
  # Take convolutions
  to_return <- dynConvolve(coeff_list)
  
  # Post-multiply (element-wise) by (k-1)!m!/(pm+k-1)!
  to_return <- mul.bigq(div.bigq(mul.bigq(as.bigq(gmp::factorialZ(k-1)), as.bigq(gmp::factorialZ(0:m))), 
                                 gmp::factorialZ(sub.bigz(add.bigz(mul.bigz(p,0:m), k),1))),
                        to_return)
  
  # Return
  return(to_return)
}

#' Get Theoretical Maximum and Minimum of Test Statistic
#' 
#' Given parameters w_vec and p, computes the model-free
#' upper and lower bounds of the test statistic. This
#' allows for more precise approximation of CDFs, in case
#' tail behaviour of approximated CDFs behave erratically.
#' (Note such erratic behaviour was observed for Jacobi approximation.)
#' 
#' Dependencies: None
#' @param w_vec The vector of weights \eqn{\boldsymbol{w}} (should be of length \eqn{k})
#' @param p Power index used in test statistic (scalar)
#' @return A vector with the minimum and the maximum values of \eqn{||S_k||_{p,\boldsymbol{w}}^p}
#' @examples 
#' getExtrema(2,rep(1,5))
#'
getExtrema <- function(p, w_vec) {
  to_return <- c(0,1)
  names(to_return) <- c("min", "max")
  
  # assign max value
  to_return[2] <- max(w_vec)
  
  # assign min value
  den <- sum(1/(p*w_vec)^(1/(p-1)))^(-p) 
  num <- 1/p^(p/(p-1)) * sum(1/w_vec^(1/(p-1)))
  to_return[1] <- den * num
  
  # return
  return(to_return)
}

#' Bernstein Approximation \eqn{p}-value
#'
#' Given the value of the test statistic, the number of moments used for Bernstein approximation, 
#' the choice of \eqn{p}, the list of moments up to the desired number of moments, and
#' the choice of alternative (two-sided or one-sided), function computes the approximate p-value
#' according to Feller's difference formula.
#'  
#' Dependencies: getExtrema
#' @param t Value of test statistic \eqn{||S_{n,k}(D)/n||_{p,\boldsymbol{w}}^p} computed from data \eqn{D}
#' @param n_mom Number of moments to use (typically \eqn{\geqslant 100})
#' @param p Power index chosen, used to normalize \eqn{S_{n,k}} to \eqn{S_k} 
#' @param k Number of bins 
#' @param moment_seq Moment sequence computed from discreteMoments
#' @param alternative Character string that should be either `two.sided` or `one.sided` (default is `two.sided`)
#' @param wList The vector of weights (should have length \eqn{k})
#' @return p-value (scalar)
#' @examples
#' 
#' getBernsteinPValue(t = 0.5, n_mom = 50, p = 2, k = 5, moment_seq = discreteMoments(m = 50, n = 6,5))
#' getBernsteinPValue(t = 0.2, n_mom = 100, p = 2, k = 5, moment_seq = continuousMoments(100,2,5))
#' 
getBernsteinPValue <- function(t, n_mom, p, k, moment_seq, alternative = "two.sided", wList = rep(1,k)) {
  # assert number of moments is equal to length of moment sequence - 1
  assertthat::assert_that(length(moment_seq) == n_mom + 1,
                          msg = "Length of moment sequence should be one plus no. moments.")
  
  if (!identical(wList, rep(1,k))) {
    assertthat::assert_that(length(wList) == k,
                            msg = "Weight vector must have length k.")
    
    assertthat::assert_that(!any(wList <0),
                            msg = "Weight vector must have non-negative weights.")
    
    assertthat::assert_that(max(wList) == 1,
                            msg = "Maximum weight should be 1.")
  }
  # compute difference vector once and cache
  summands <- sapply(1:n_mom, 
                     function(y) {
                       as.numeric(gmp::mul.bigq(chooseZ(n_mom,y-1), 
                                                diff(moment_seq, differences = n_mom - y + 1)[y]) * 
                                    (-1)^(n_mom - y + 1) )}) 
  
  # compute model-free extrema of test statistic 
  extrema <- getExtrema(p, wList)
  #arg_modes <- seq(extrema[1], extrema[2], length.out = mesh)
  
  getCDF <- function(x) {
    to_include <- sapply(0:(n_mom-1), function(y) {ifelse(x >= y/n_mom, 1, 0)}) 
    return(sum(to_include * summands))
  }
  
  if (t < extrema[1] | t > extrema[2]) {
    warning("The test statistic value exceeds model-free bounds for the test statistic.")
    message(paste0("Theoretical maximum = ", extrema[2], "; theoretical minimum = ", extrema[1], "; observed statistic = ", t))
    return(0)
  }  else  {
    to_return <- getCDF(t) 
    #assertthat::assert_that(to_return <= 1 & to_return >=0, 
    #                         msg = "Bernstein approximation is inaccurate.")
    if (alternative == "two.sided") {
      return(2 * min(to_return, 1-to_return))  
    } else {
      return(min(to_return, 1-to_return))
    }
  }
}

#' Chebyshev and Jacobi Approximation \eqn{p}-value
#' 
#' Given the value of the test statistic, the number of moments used for approximation, 
#' the list of moments up to the desired number of moments, the method of approximation
#' by moments, and the choice of alternative (two-sided or one-sided), function computes 
#' the approximate p-value according to an inner product approximation.
#'  
#' Dependencies: None
#' @param t Value of test statistic \eqn{||S_{n,k}(D)/n||_{p,\boldsymbol{w}}^p} computed from data \eqn{D}
#' @param n_mom Number of moments to use (typically \eqn{\geqslant 50})
#' @param moment_seq Moment sequence computed from discreteMoments
#' @param method Character string that should be either `chebyshev` or `jacobi` (default is `chebyshev`)
#' @param alternative Character string that should be either `two.sided` or `one.sided` (default is `two.sided`)
#' @return p-value (scalar)
#' @examples
#' 
#' getMomentPValue(t = 0.5, n_mom = 50, moment_seq = discreteMoments(m = 50, n = 6,5))
#' getMomentPValue(t = 0.2,n_mom = 100, moment_seq = continuousMoments(100,2,5))
#' 
getMomentPValue <- function(t, n_mom, moment_seq, method = "chebyshev", alternative = "two.sided") {
  assertthat::assert_that((method == "chebyshev" | method == "jacobi"),
                          msg = "Please choose either `chebyshev` or `jacobi` as approximation method.")
  
  if (method == "chebyshev") {
    moment_seq_trunc <- moment_seq[1:n_mom]
  } else {
    moment_seq_trunc <- moment_seq[1:(4*floor(n_mom/4) + 1)] 
  }
  
  # find closest value in pre-computed basket of interpolating polynomials
  closest_index <- which(abs(t-seq(from=0, to = 0.998, by = 0.002)) == min(abs(t-seq(from=0, to = 0.998, by = 0.002))))[1]
  dec <- seq(from=0, to = 0.998, by = 0.002)[closest_index]
  
  # retrieve Lagrange coefficients for closest value
  lagrange_poly_coefs <- read.table(paste0("~/Documents/research/spacing_stats/010422/",
                                           method,"/m", 
                                           n_mom, 
                                           "/coeff_list_m", 
                                           n_mom, "_", dec, ".txt"), 
                                    numerals = "no.loss")
  lagrange_poly_precise_coefs <- Rmpfr::mpfr(lagrange_poly_coefs$V1, 5000)
  
  # compute inner product
  precise_moments <- Rmpfr::mpfr(moment_seq_trunc, 5000)
  to_return <- as.numeric(sum(precise_moments * lagrange_poly_precise_coefs))
  #print(to_return)
  #assertthat::assert_that(to_return <= 1 & to_return >=0, 
  #                        msg = paste0(method, " approximation is inaccurate."))
  if (alternative == "two.sided") {
    return(2 * min(to_return, 1-to_return))  
  } else {
    return(min(to_return, 1-to_return))
  }
}

#' Approximate \eqn{p}-value by Resampling Integer Compositions
#' 
#' Given the value of the test statistic \eqn{t}, the sample sizes \eqn{n} and \eqn{k}, 
#' power exponent \eqn{p} and vector of weights that together determine the test statistic 
#' (by default \eqn{n\geqslant k}), as well as the user-specified resampling number 
#' (by default this is \eqn{5000}), performs resampling from the collection of integer compositions 
#' to approximate the \eqn{p}-value of the observed test statistic.
#' 
#' The function returns a two-sided \eqn{p}-value, which is more conservative. Users can choose other
#' \eqn{p}-values corresponding to different alternatives; see documentation on `alternative` below.
#' Note that the interpretation of the choice of `alternative` depends on the choice of weight vector.
#' For example, a weight vector that is a quadratic kernel will upweight the extreme components of 
#' the weight vector. For this choice, setting `alternative` to be `bigger` translates into an alternative 
#' hypothesis of a bigger spread in the larger sample (the one with sample size \eqn{n}).   
#'  
#' Dependencies: arrangements::compositions
#' @param t Value of test statistic \eqn{||S_{n,k}(D)/n||_{p,\boldsymbol{w}}^p} computed from data \eqn{D}
#' @param n Sample size of \eqn{y}
#' @param k Sample size of \eqn{x}
#' @param p Power exponent of test statistic
#' @param wList Weight vector
#' @param alternative Character string that should be one of `two.sided`, `bigger` or `smaller` (default is `two.sided`)
#' @return p-value (scalar)
#' @examples
#' 
#' getCompositionPValue(t = 0.5, n = 50, k = 11, p = 1, wList = (10:0)/10)
#' getCompositionPValue(t = 0.2,n_mom = 100, moment_seq = continuousMoments(100,2,5))
#'
getCompositionPValue <- function(t, n, k, p, wList, alternative = "two.sided", resamp_number = 5000) {
  # Make sure that alternative is well-defined
  assertthat::assert_that((alternative == "two.sided" | alternative == "bigger" | alternative == "smaller"), 
                          msg = "Please specify a valid alternative (two.sided, bigger, or smaller)")
  
  # Sample test statistic and compute empirical CDF at t
  resampled_ts <- ((compositions(n = n, k = k, nsample = resamp_number)/n)^p %*% wList) %>% as.vector()
  cdf_at_t <- mean(resampled_ts < t) 
  
  # Compute p-value
  if (alternative == "two.sided") {
    message(date(), ": Computing two-sided p-value")
    return(2*min(cdf_at_t, 1 - cdf_at_t))
    
  } else if (alternative == "bigger") {
    message(date(), ": Computing one-sided p-value with alternative set to bigger")
    return(cdf_at_t)
    
  } else {
    message(date(), ": Computing one-sided p-value with alternative set to smaller")
    return(1 - cdf_at_t)
  }
}

#' Resampling a test statistic from the continuous simplex
#' 
#' Given the vector of weights \eqn{\boldsymbol{w}} and exponent \eqn{p} ---
#' which determine the test statistic \eqn{||S_{k}||_{p,\boldsymbol{w}}^p} entirely --- as well as
#' the dimensionality \eqn{k} (determined by the the sample size of the data), draws a 
#' single observation from the uniform distribution on the continuous \eqn{k-1}-dimensional
#' simplex. 
#' 
#' Dependencies: None
#' @param w_vec Vector of weights 
#' @param p Exponent value in defining test statistic (must be integer)
#' @param k Dimensionality for simplex (typically equal to one less than the sample size of the one sample \eqn{x})
#' @return A single observation of the test statistic from the uniform distribution on the continuous simplex
#' @examples
#' 
#' getOneTestStat(w_vec = rep(1,10), p = 2, k = 10)
#' 
getOneTestStat <- function(w_vec, 
                           p, 
                           k) {
  s_vec <- simplex.sample(n=k, N=1, sort=FALSE)$samples %>% as.vector() 
  return(sum(w_vec * s_vec^p))
}
  
## Some simple tests 
#p_value <- getBernsteinPValue(t = 0.5, n_mom = 50, p = 2, k = 5, moment_seq = discreteMoments(m = 50, n = 6,5))
#cdf_df  <- getBernsteinCDF2(50,2,moment_seq = discreteMoments(m = 50, n = 6,5), mesh = 500)
#data.frame(x = seq(0,1,length.out=500),
#           y = cdf_df)
# p_value_2 <- getMomentPValue(t = 0.5, n_mom = 100, moment_seq = discreteMoments(m = 100, n = 25,5))
# # [1] 0.9795116
# p_value_3 <- getMomentPValue(t = 0.5, n_mom = 100, moment_seq = continuousMoments(m=100,p = 2,k=5))

