######################################################
######## Main Function for Computing p-values ########
############## Edited on Mar 24, 2022 ################
######################################################

# Load functions in back-end R script (change dir if necessary)
source("~/Documents/research/spacing_stats/012522/local_functions.R")
# Load functions written in Python (change dir if necessary)
#reticulate::source_python("local_functions_no_ray.py")
#reticulate::source_python("../Python/local_functions.py")
# [!] To do: automate installation of conda env with required Python modules

#' Flexible Non-Parametric One- and Two-Sample Tests (Native R version)
#' 
#' Given data consisting of either a single sample \eqn{\boldsymbol{x}=(x_1,\ldots,x_k)}, or two samples
#' \eqn{\boldsymbol{x}=(x_1,\ldots,x_k)} and \eqn{\boldsymbol{y}=(y_1,\ldots,y_n)}, this function uses summary statistics 
#' computed on weighted linear combinations of powers of the spacing statistics \eqn{S_k} (former) 
#' or \eqn{S_{n,k}} (latter). More precisely, this function does the following.
#' 
#' For a single sample \eqn{x}, the function tests for uniformity of its entries. When \eqn{p=2},
#' the test is just Greenwood's test. 
#' 
#' For two samples, the function tests the null of \eqn{x} and \eqn{y} being drawn from the 
#' same distribution, against flexible alternatives that correspond to specific choices of 
#' the test statistic parameters, \eqn{\boldsymbol{w}} (weight vector) and \eqn{p} (power). 
#' More precisely, these parameters not only determine the test statistic 
#' \eqn{||S_k||_{p,\boldsymbol{w}}^p=\sum_{j=1}^k w_iS_{k}[j]^p} (analogously defined for 
#' \eqn{||S_{n,k}||_{p,\boldsymbol{w}}^p}), but also encode alternative hypotheses
#' ranging from different populational means (i.e., \eqn{\mu_x \neq \mu_y}), different 
#' populational spreads (i.e., \eqn{\sigma^2_x \neq \sigma^2_y}), etc. 
#' 
#' Additional tuning parameters include (1) choice of p-value computation (one- or two-sided);
#' (2) approximation method (Bernstein, Chebyshev or Jacobi); (3) number of moments accompanying the 
#' approximation chosen (recommended 200, typically at least 100); and (4) in case of two samples,
#' whether the user prefers to use exact discrete moments (more accurate but slower) or to use
#' continuous approximations of the discrete moments (less accurate but faster).         
#' 
#' Dependencies: functions in local_functions.R
#' @param x First sample
#' @param y Second sample (default is NULL)
#' @param p Exponent value in defining test statistic (must be integer for continuous moments)
#' @param wList Vector of weights. It should always have length one more than the length of \eqn{x}
#' @alternative How p-value should be computed (choose `one.sided` or `two.sided`)
#' @approx Which approximation method to use (choose `bernstein`, `chebyshev` or `jacobi`)
#' @n_mom The number of moments to accompany the approximation (recommended 200, if not at least 100)
#' @force_discrete In the two-sample case, whether to use discrete moments even if \eqn{n} is large enough (default is FALSE)
#' @examples 
#' 
#' # One-sample examples
#' mochisR(x = abs(rnorm(10)), p = 2, wList = rep(1,10), alternative = "two.sided", approx = "chebyshev", n_mom = 200) # good calibration
#' mochisR(x = abs(rnorm(10)), p = 2, wList = rep(1,10), alternative = "two.sided", approx = "bernstein", n_mom = 200) # false negatives
#' mochisR(x = abs(rnorm(10)), p = 2, wList = rep(1,10), alternative = "two.sided", approx = "jacobi", n_mom = 100) 
#' 
#' # Two-sample examples
#' mochisR(x = abs(rnorm(10)), y = abs(rnorm(100)), p = 2, wList = rep(1,10), alternative = "two.sided", approx = "chebyshev", n_mom = 200) # good calibration
#' mochisR(x = abs(rnorm(10)), y = abs(rnorm(100)), p = 2, wList = rep(1,10), alternative = "two.sided", approx = "bernstein", n_mom = 200) # good calibration
#' mochisR(x = abs(rnorm(10)), y = abs(rnorm(100)), p = 2, wList = rep(1,10), alternative = "two.sided", approx = "jacobi", n_mom = 200) # good calibration
#' 
mochisR <- function(x, y = NULL, 
                    p, 
                    wList, 
                    alternative, 
                    approx, 
                    n_mom, 
                    force_discrete = FALSE) {
  # 1. Get number of bins
  k <- length(x) + 1
  
  # 2. Normalise weights
  message(date(), ": Normalizing weight vector...")
  wList <- wList / max(wList)
  
  # 3. Compute test statistic t and return its p-value
  # 3.1. Case 1: y is not NULL
  if (!is.null(y)) {
    # construct ordering of x_i's
    x_ordered <- x[order(x)]
    x_ordered <- c(-Inf, x_ordered, Inf) 
    n <- length(y) # get sample size / number of balls
    
    # construct Snk
    Snk <- c()
    for (i in 1:k) {
      # count number of y_j's between x_i and x_{i+1}
      Snk <- c(Snk, sum(y >= x_ordered[i] &  y < x_ordered[i+1]))
    }
    
    # construct t
    t <- sum((Snk/n)^p * wList)
    message(date(), ": The test statistic for the data is ", t)
    
    # decide whether to use large n asymptotics or not
    if (n >= 100 & k >= 50 & k/n >= 1e-3 & (p == 1 | p == 2)) {
      message(
        date(),
        ": Sample sizes, n and k, large enough such that k/n > 0; p = 1 or p = 2. Applying Gaussian asymptotics...")
      
      # [!] 3/27/22 Update: Jonathan and I discovered poor FPR using the Monte Carlo method.
      # Changing the implementation to rely on Dan's analytical formulae.
      
      # Compute analytical mean and variance
      if (p == 1) {
        # p = 1 
        first_moment <- sum(wList) / k
        second_moment <- ((k/n+1)/(k^2*(k+1))) * 
          sum(wList *((k*diag(k) - outer(rep(1,k),rep(1,k))) %*% wList))
        #second_moment <- ((k/n+1) / (k^2*(k+1))) * sum(wList*sum(wList)*(k-1)) 
        #second_moment <- ((k/n+1) / (k^2*(k+1))) * sum(wList*sum(wList)*(k-1)) 
      } else {
        # p = 2
        first_moment <- ((2+k/n-1/n) / ((k+1)*k)) * sum(wList)
        sum_of_wj2s <- sum(wList^2)
        coeff_sum_of_wj2s <- (k-1) * (k/n+1) * (2+k/n-1/n) * (12-6/n+k*(k/n+10-5/n)) / (k^2*(1+k)^2*(2+k)*(3+k))
        offdiag_sum <- sum(outer(wList, wList)) - sum(wList^2)
        coeff_offdiag_sum <- (k/n+1) * (6/n^2+k*(3+k*(k-2))/n^2-24/n+8*(k-1)*k/n+8*(3+2*k)) / (k^2*(1+k)^2*(2+k)*(3+k))
        second_moment <- sum_of_wj2s * coeff_sum_of_wj2s - offdiag_sum * coeff_offdiag_sum
      }
      
      z_score <- (t - first_moment) / sqrt(second_moment)
      
      ## Currently not computing the Gaussian mean and sd, using Monte Carlo instead
      #simplex_samples <- replicate(10000, expr = getOneTestStat(w_vec = wList, 
      #                                                          p = p, 
      #                                                          k = k))
      #z_score <- (t - mean(simplex_samples)) / sd(simplex_samples)
      if (alternative == "one.sided") {
        return(min(pnorm(z_score), pnorm(-z_score)))
        #return(min(mean(simplex_samples < t), 1 - mean(simplex_samples < t)))
      } else {
        return(2*min(pnorm(z_score), pnorm(-z_score)))
        #return(2 * min(mean(simplex_samples < t), 1 - mean(simplex_samples < t)))
      }
    } else if (n >= 100 & !force_discrete) {
      message(date(), ": Sample size, n, is large enough, using Sk distribution...")
      
      # compute continuous moments
      message(date(), ": Computing continuous moments...")
      moment_seq <- continuousMoments(m = n_mom, p = p, k = k, wList = wList)
      
      # compute and return p-value
      if (approx == "bernstein") {
        return(getBernsteinPValue(t = t, 
                                  n_mom = n_mom, 
                                  p = p, k = k, 
                                  moment_seq = moment_seq,
                                  alternative = alternative,
                                  wList = wList))
      } else {
        return(getMomentPValue(t = t, 
                               n_mom = n_mom, 
                               moment_seq = moment_seq,
                               method = approx,
                               alternative = alternative))
      }
    } else {
      message(date(), ": Using Snk distribution...")
      
      # compute discrete moments
      message(date(), ": Computing discrete moments...")
      moment_seq <- discreteMoments(m = n_mom, n = n, k = k, p = p, wList = wList)
      
      # compute and return p-value
      if (approx == "bernstein") {
        return(getBernsteinPValue(t = t, 
                                  n_mom = n_mom, 
                                  p = p, k = k, 
                                  moment_seq = moment_seq,
                                  alternative = alternative,
                                  wList = wList))
      } else {
        return(getMomentPValue(t = t, 
                               n_mom = n_mom, 
                               moment_seq = moment_seq,
                               method = approx,
                               alternative = alternative))
      }
    }
    
  } else {
    # 3.2. Case 2: y is NULL => use continuous moments
    # check all entries of observations are non-negative
    assertthat::assert_that(!any(x < 0),
                            msg = "Observations should be non-negative")
    # construct Sk
    Sk <- x / sum(x)
    
    # construct t
    t <- sum(Sk^p * wList)
    
    # compute moments
    moment_seq <- continuousMoments(m = n_mom, p = p, k = k, wList = wList)
    
    # compute and return p-value
    if (approx == "bernstein") {
      return(getBernsteinPValue(t = t, 
                                n_mom = n_mom, 
                                p = p, k = k, 
                                moment_seq = moment_seq,
                                alternative = alternative,
                                wList = wList))
    } else {
      return(getMomentPValue(t = t, 
                             n_mom = n_mom, 
                             moment_seq = moment_seq,
                             method = approx,
                             alternative = alternative))
    }
  }
}

#' Flexible Non-Parametric One- and Two-Sample Tests (Python back-end version)
#' 
#' Given data consisting of either a single sample \eqn{\boldsymbol{x}=(x_1,\ldots,x_k)}, or two samples
#' \eqn{\boldsymbol{x}=(x_1,\ldots,x_k)} and \eqn{\boldsymbol{y}=(y_1,\ldots,y_n)}, this function uses summary statistics 
#' computed on weighted linear combinations of powers of the spacing statistics \eqn{S_k} (former) 
#' or \eqn{S_{n,k}} (latter). More precisely, this function does the following.
#' 
#' For a single sample \eqn{x}, the function tests for uniformity of its entries. When \eqn{p=2},
#' the test is just Greenwood's test. 
#' 
#' For two samples, the function tests the null of \eqn{x} and \eqn{y} being drawn from the 
#' same distribution, against flexible alternatives that correspond to specific choices of 
#' the test statistic parameters, \eqn{\boldsymbol{w}} (weight vector) and \eqn{p} (power). 
#' More precisely, these parameters not only determine the test statistic 
#' \eqn{||S_k||_{p,\boldsymbol{w}}^p=\sum_{j=1}^k w_iS_{k}[j]^p} (analogously defined for 
#' \eqn{||S_{n,k}||_{p,\boldsymbol{w}}^p}), but also encode alternative hypotheses
#' ranging from different populational means (i.e., \eqn{\mu_x \neq \mu_y}), different 
#' populational spreads (i.e., \eqn{\sigma^2_x \neq \sigma^2_y}), etc. 
#' 
#' Additional tuning parameters include (1) choice of p-value computation (one- or two-sided);
#' (2) approximation method (Bernstein, Chebyshev or Jacobi); (3) number of moments accompanying the 
#' approximation chosen (recommended 200, typically at least 100); and (4) in case of two samples,
#' whether the user prefers to use exact discrete moments (more accurate but slower) or to use
#' continuous approximations of the discrete moments (less accurate but faster).         
#' 
#' Dependencies: functions in local_functions.py
#' @param x First sample
#' @param y Second sample (default is NULL)
#' @param p Exponent value in defining test statistic (must be integer for continuous moments)
#' @param wList Vector of weights. It should always have length one more than the length of \eqn{x}
#' @alternative How p-value should be computed (choose `one.sided` or `two.sided`)
#' @approx Which approximation method to use (choose `bernstein`, `chebyshev` or `jacobi`)
#' @n_mom The number of moments to accompany the approximation (recommended 200, if not at least 100)
#' @force_discrete In the two-sample case, whether to use discrete moments even if \eqn{n} is large enough (default is FALSE)
#' @examples 
#' 
#' # One-sample examples
#' mochisPy(x = abs(rnorm(10)), p = 2, wList = rep(1,10), alternative = "two.sided", approx = "chebyshev", n_mom = 200) # good calibration
#' mochisPy(x = abs(rnorm(10)), p = 2, wList = rep(1,10), alternative = "two.sided", approx = "bernstein", n_mom = 200) # false negatives
#' mochisPy(x = abs(rnorm(10)), p = 2, wList = rep(1,10), alternative = "two.sided", approx = "jacobi", n_mom = 100) 
#' 
#' # Two-sample examples
#' mochisPy(x = abs(rnorm(10)), y = abs(rnorm(100)), p = 2, wList = rep(1,10), alternative = "two.sided", approx = "chebyshev", n_mom = 200) # good calibration
#' mochisPy(x = abs(rnorm(10)), y = abs(rnorm(100)), p = 2, wList = rep(1,10), alternative = "two.sided", approx = "bernstein", n_mom = 200) # good calibration
#' mochisPy(x = abs(rnorm(10)), y = abs(rnorm(100)), p = 2, wList = rep(1,10), alternative = "two.sided", approx = "jacobi", n_mom = 200) # good calibration
#'   
mochisPy <- function(x, y = NULL, 
                     p, 
                     wList, 
                     alternative, 
                     approx, 
                     n_mom, 
                     force_discrete = FALSE) {
  # 1. Get number of bins
  k <- length(x) + 1
  
  # 2. Normalise weights
  message(date(), ": Normalizing weight vector...")
  wList <- wList / max(wList)
  
  # 3. Compute test statistic t and return its p-value
  # 3.1. Case 1: y is not NULL
  if (!is.null(y)) {
    # construct ordering of x_i's
    x_ordered <- x[order(x)]
    x_ordered <- c(-Inf, x_ordered, Inf) 
    n <- length(y) # get sample size / number of balls
    
    # construct Snk
    Snk <- c()
    for (i in 1:k) {
      # count number of y_j's between x_i and x_{i+1}
      Snk <- c(Snk, sum(y >= x_ordered[i] &  y < x_ordered[i+1]))
    }
    
    # construct t
    t <- sum((Snk/n)^p * wList)
    message(date(), ": The test statistic for the data is ", t)
    
    # decide whether to use large n asymptotics or not
    if (n >= 100 & !force_discrete) {
      message(date(), ": Sample size, n, is large enough, using Sk distribution...")
      
      # compute continuous moments
      message(date(), ": Computing continuous moments...")
      moment_seq <- continuous_moments(m=as.integer(n_mom),
                                       p=as.integer(p),
                                       k=as.integer(k),
                                       wList=wList)
      
      # compute and return p-value
      if (approx == "bernstein") {
        return(get_Bernstein_pvalue(t = t, 
                                  n_mom = as.integer(n_mom), 
                                  p = p, k = as.integer(k), 
                                  moment_seq = moment_seq,
                                  alternative = alternative,
                                  wList = wList))
      } else {
        return(get_moment_pvalue(t = t, 
                               n_mom = as.integer(n_mom), 
                               moment_seq = moment_seq,
                               method = approx,
                               alternative = alternative))
      }
    } else {
      message(date(), ": Using Snk distribution...")
      
      # compute discrete moments
      message(date(), ": Computing discrete moments...")
      moment_seq <- discrete_moments(m = as.integer(n_mom), 
                                     n = as.integer(n), 
                                     #k = as.integer(k), p = p,
                                     k = as.integer(k), p = as.integer(p), # somehow fixes the overflow
                                     wList = wList)
      
      # compute and return p-value
      if (approx == "bernstein") {
        return(get_Bernstein_pvalue(t = t, 
                                  n_mom = as.integer(n_mom), 
                                  p = p, k = as.integer(k), 
                                  moment_seq = moment_seq,
                                  alternative = alternative,
                                  wList = wList))
      } else {
        return(get_moment_pvalue(t = t, 
                               n_mom = as.integer(n_mom), 
                               moment_seq = moment_seq,
                               method = approx,
                               alternative = alternative))
      }
    }
    
  } else {
    # 3.2. Case 2: y is NULL => use continuous moments
    # check all entries of observations are non-negative
    assertthat::assert_that(!any(x < 0),
                            msg = "Observations should be non-negative")
    # construct Sk
    Sk <- x / sum(x)
    
    # construct t
    t <- sum(Sk^p * wList)
    
    # compute moments
    moment_seq <- continuous_moments(m=as.integer(n_mom),
                                     p=as.integer(p),
                                     k=as.integer(k),
                                     wList=wList)
    
    # compute and return p-value
    if (approx == "bernstein") {
      return(get_Bernstein_pvalue(t = t, 
                                n_mom = as.integer(n_mom), 
                                p = p, k = as.integer(k), 
                                moment_seq = moment_seq,
                                alternative = alternative,
                                wList = wList))
    } else {
      return(get_moment_pvalue(t = t, 
                             n_mom = as.integer(n_mom), 
                             moment_seq = moment_seq,
                             method = approx,
                             alternative = alternative))
    }
  }
}

#' MOCHIS
#' 
#' Given data consisting of either a single sample \eqn{\boldsymbol{x}=(x_1,\ldots,x_k)}, or two samples
#' \eqn{\boldsymbol{x}=(x_1,\ldots,x_k)} and \eqn{\boldsymbol{y}=(y_1,\ldots,y_n)}, this function uses summary statistics 
#' computed on weighted linear combinations of powers of the spacing statistics \eqn{S_k} (former) 
#' or \eqn{S_{n,k}} (latter). More precisely, this function does the following.
#' 
#' For a single sample \eqn{x}, the function tests for uniformity of its entries. When \eqn{p=2},
#' the test is just Greenwood's test. 
#' 
#' For two samples, the function tests the null of \eqn{x} and \eqn{y} being drawn from the 
#' same distribution, against flexible alternatives that correspond to specific choices of 
#' the test statistic parameters, \eqn{\boldsymbol{w}} (weight vector) and \eqn{p} (power). 
#' More precisely, these parameters not only determine the test statistic 
#' \eqn{||S_k||_{p,\boldsymbol{w}}^p=\sum_{j=1}^k w_iS_{k}[j]^p} (analogously defined for 
#' \eqn{||S_{n,k}||_{p,\boldsymbol{w}}^p}), but also encode alternative hypotheses
#' ranging from different populational means (i.e., \eqn{\mu_x \neq \mu_y}), different 
#' populational spreads (i.e., \eqn{\sigma^2_x \neq \sigma^2_y}), etc. 
#' 
#' Additional tuning parameters include (1) choice of Python or R backend  (Python is much faster but requires 
#' Python installation); (2) choice of p-value computation (one- or two-sided); (3) approximation method 
#' (Bernstein, Chebyshev or Jacobi); (4) number of moments accompanying the approximation chosen 
#' (recommended 200, typically at least 100); and (5) in case of two samples, whether the user prefers 
#' to use exact discrete moments (more accurate but slower) or to use continuous approximations of 
#' the discrete moments (less accurate but faster).         
#' 
#' Dependencies: mochisPy, mochisR
#' @param x First sample
#' @param y Second sample (default is NULL)
#' @param p Exponent value in defining test statistic (must be integer for continuous moments)
#' @param wList Vector of weights. It should always have length one more than the length of \eqn{x}
#' @alternative How p-value should be computed (choose `one.sided` or `two.sided`)
#' @approx Which approximation method to use (choose `bernstein`, `chebyshev` or `jacobi`)
#' @n_mom The number of moments to accompany the approximation (recommended 200, if not at least 100)
#' @force_discrete In the two-sample case, whether to use discrete moments even if \eqn{n} is large enough (default is FALSE)
#' @python_backend To use Python or R backend, with `TRUE` meaning that a call to Pythonic functions will occur (default is TRUE)
#' @examples 
#' 
#' # One-sample examples
#' mochis.test(x = abs(rnorm(10)), p = 2, wList = rep(1,10), alternative = "two.sided", approx = "chebyshev", n_mom = 200) # good calibration
#' 
#' # Two-sample examples
#' mochis.test(x = abs(rnorm(10)), y = abs(rnorm(100)), p = 2, wList = rep(1,10), alternative = "two.sided", approx = "chebyshev", n_mom = 200) # good calibration
#' 
mochis.test <- function(x, y = NULL, 
                        p, 
                        wList, 
                        alternative, 
                        approx, 
                        n_mom, 
                        force_discrete = FALSE,
                        python_backend = TRUE) {
  # if Python, then use mochisPy
  # [!] TO ADD AUTO-DETECTION OF PYTHON3 COMPATIBILITY 
  if (python_backend) {
    message(date(), ": Using Python3 back-end to compute p-value...")
    return(mochisPy(x, y, p, wList, alternative, approx, n_mom, force_discrete))
  } else {
    # else use mochisR
    message(date(), ": Using native R to compute p-value...")
    return(mochisR(x, y, p, wList, alternative, approx, n_mom, force_discrete))
  }
}

# set.seed(2022)
# x0 <- abs(rnorm(10)); y0 <- abs(rnorm(100))
# mochis.test(x = x0,
#             y = y0,
#             p = 2, wList = rep(1,10),
#             alternative = "two.sided",
#             approx = "chebyshev", n_mom = 200,
#             python_backend = TRUE)
#
# mochis.test(x = x0,
#             y = y0,
#             p = 2, wList = rep(1,10),
#             alternative = "two.sided",
#             approx = "bernstein", n_mom = 200,
#             python_backend = TRUE)

# x1 <- abs(rnorm(10)); y1 <- abs(rnorm(50))
# # ERROR: result is too large
# # ERROR is fixed by changing p to an integer
# mochis.test(x = x1,
#             y = y1,
#             p = 2, wList = rep(1,10),
#             alternative = "two.sided",
#             approx = "bernstein", n_mom = 150,
#             python_backend = TRUE)

# # ERROR: 2022-02-19 21:54:17,581	ERROR services.py:1254 -- Failed to start the dashboard:
# # Failed to read dashbord log: [Errno 2] No such file or directory:
# # ERROR is fixed by changing p to an integer
# # '/tmp/ray/session_2022-02-19_21-54-16_009127_37276/logs/dashboard.log'
# mochis.test(x = x1,
#             y = y1,
#             p = 2, wList = rep(1,10),
#             alternative = "two.sided",
#             approx = "bernstein", n_mom = 50,
#             python_backend = TRUE)
# Make sure reticulate is using a venv that has Python3.8.5 (to match Rachel's)
