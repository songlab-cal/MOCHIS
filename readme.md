# MOment Computation and Hypothesis testing Induced by S<sub>n,k</sub> (MOCHIS)

This is the repository for organizing and distributing code and results related to the
family of spacing-statistics and their associated hypothesis test detailed in:

Erdmann-Pham, D.D., Terhorst, J., and Song, Y.S. 
<em>Generalized Spacing-Statistics and a New Family of Non-Parametric Tests.</em>
  Preprint: https://arxiv.org/abs/2008.06664

## Code

### mochis.nb

This is a Mathematica notebook to reproduce results and figures used in our main
manuscript, as well as to perform general one-sample and two sample tests. It contains functions for 

* computing the moment sequences associated with the weighted p-norms of S<sub>n,k</sub> and S<sub>k</sub>
* accurate evaluation of a distribution's CDF given its first m moments
* performing one- and two-sample tests
* optimizing parameter configurations (p, w) for power against any given set of alternative distributions
