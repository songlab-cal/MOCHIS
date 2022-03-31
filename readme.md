# MOCHIS :dango:

Welcome to MOCHIS (MOment Computation and Hypothesis testing Induced by S<sub>n,k</sub>), software for implementing flexible and exact two-sample tests, with applications to single cell genomics. It is based on the work of [Erdmann-Pham et al. (2022+)](https://arxiv.org/abs/2008.06664).

This is the **development branch**, with work still being in progress. There are three subdirectories:
- `/Mathematica` contains a [Mathematica](https://www.wolfram.com/mathematica/) implementation of the methods and examples reproducing key results in the paper 
- `/R` contains [R](https://www.r-project.org/about.html) implementation of the methods and application to single cell RNA-seq data
- `/Python` contains [Python](https://www.python.org/) implementation of the methods and application to single cell RNA-seq data

Feel free to read more below.

## Methodology

MOCHIS implements tests based on summary statistics computed from points lying in the (discrete or continuous) _k_-dimensional simplex. These tests include:

- the [Mann-Whitney](https://www.sciencedirect.com/topics/medicine-and-dentistry/rank-sum-test) (Wilcoxon rank sum) test for two samples;
- [Greenwood's](https://en.wikipedia.org/wiki/Greenwood_statistic) test of randomness; 

they also include tests designed to maximize power for a range of alternative hypotheses, including differences in spread and differences in location and scale.

Please see `theory_behind_implementation.pdf`, a manual explaining our software. 

## Contributors

- Alan Aw (Statistics PhD student, UC Berkeley)
- Xurui (Rachel) Chen (Undergraduate in Computer Science and Statistics, UC Berkeley)
- Đan-Daniel Erdmann-Pham (Stein Fellow in Statistics, Stanford University)
- Jonathan Fischer (Clinical Assistant Professor in Biostatistics, University of Florida)
- Yun S. Song (Professor of EECS and Statistics, UC Berkeley)