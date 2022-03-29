# R User Manual

Our software offers an optional interface with Python. All steps relevant to setting up this interface are marked by the python emoji (:snake:), and users who do not have Python may choose to skip them. 

:snake: Install Python 3.7 or a more up-to-date version (see how to do so [here](https://realpython.com/installing-python/)). While entirely optional, we recommend doing so because our main function has an option of interfacing with Python, which speeds up computation significantly. 

1. (Download required files) Download this directory. Also download the following zip files.
    - `chebyshev.zip` ([link](https://www.dropbox.com/s/t2ml80d3pl6p83q/chebyshev.zip?dl=0))
    - `jacobi.zip` ([link](https://www.dropbox.com/s/2envgy7q99ntkmj/jacobi.zip?dl=0))

    :snake: Go to the sibling directory of this directory, named `python`, and download the two files

    - `local_functions_for_R.py`
    - `local_functions.py` 

    Create a sibling directory `python` and place these Python files under it. 

2. (Install package dependencies) Install the following packages in R if you don't already have them installed. 
    - [**gmp**](https://cran.r-project.org/web/packages/gmp/)
    - [**doParallel**](https://cran.r-project.org/web/packages/doParallel/)
    - [**Rmpfr**](https://cran.r-project.org/web/packages/Rmpfr/)
    - [**assertthat**](https://cran.r-project.org/web/packages/assertthat/)
    - :snake: [**reticulate**](https://cran.r-project.org/web/packages/reticulate/) 

    If you are already familiar with R or RStudio, you can install these packages as you would normally do when you need to use a new package. Alternatively, you may accomplish this step by running the following line of code after starting the R session in Step 4. 

    ```
    install.packages(c("gmp", "doParallel", "Rmpfr", "assertthat"))
    ```

    :snake: For Python interfacing:

    ```
    install.packages(c("gmp", "doParallel", "Rmpfr", "assertthat", "reticulate"))
    ```

3. (Setup lookup table files) Create a directory called `chebyshev`, then unzip `chebyshev.zip` and put all folders (`m25`, `m50`, etc.) under `chebyshev`. Do the same for `jacobi`.  

    At this stage, you should have the following directory structure under your current working directory.

    - `main_draft0.R`
    - `local_functions.R`
    - `chebyshev`
        - `m25`
        - `m50`
        - `...`
    - `jacobi` 
        - `m25`
        - `m50`
        - `...`

    :snake: If using Python backend, the working directory ("`CURRENT_WORKING_DIR`") should have a sibling directory named `python` that contains the two Python scripts, as described in Step 1. In other words, one level above the working directory should have this structure.

    - `python`
        - `local_functions_for_R.py`
        - `local_functions.py`
    - `CURRENT_WORKING_DIR`
        - `...`

4. (Load functions in R) Start R/RStudio and open `main_draft0.R` in a text editor or within RStudio. 

    - Run `source("main_draft0.R")` under your current working directory. This should load all functions required for running our one- and two-sample tests. 
    - :snake: Install **gmpy2** (a Python package) using R, as shown below.

        ```
        reticulate::py_install("gmpy2")
        ```

        Uncomment Line 8 of `main_draft0.R` (which is `reticulate::source_python("../python/local_functions_for_R.py")`), and then run `source("main_draft0.R")`. This should load all functions, including Python backend functions, for running our one- and two-sample tests. This step will also reveal any Python package dependencies that you do not already have installed. Please install them within the environment that **reticulate** has set up to interface with Python. This can be accomplished by either directly installing them on the terminal (e.g., `pip install [PACKAGE-NAME]` or `conda install [PACKAGE-NAME]`) or via **reticulate** within R, as shown above.

## Examples

We simulate two samples and then show how we can run our tests on them. We set the seed for our simulation to ensure reproducibility.

```
# Simulate samples
set.seed(2022)
x0 <- abs(rnorm(10)); y0 <- abs(rnorm(100))
```

Next, run the following to compute the _p_-value.  

```
# Use native R backend
mochis.test(x = x0,
            y = y0,
            p = 2, wList = rep(1,11),
            alternative = "two.sided",
            approx = "chebyshev", n_mom = 200,
            python_backend = FALSE)
```

We obtain a p-value of 0.68.

:snake: Users relying on the Python back-end can run the following: 

```
# Use Python backend
mochis.test(x = x0,
            y = y0,
            p = 2, wList = rep(1,11),
            alternative = "two.sided",
            approx = "chebyshev", n_mom = 200,
            python_backend = TRUE)

# Fri Feb 25 16:23:22 2022: Using Python3 back-end to compute p-value...
# Fri Feb 25 16:23:22 2022: Normalizing weight vector...
# Fri Feb 25 16:23:22 2022: The test statistic for the data is 0.1462
# Fri Feb 25 16:23:22 2022: Sample size, n, is large enough, using Sk distribution...
# Fri Feb 25 16:23:22 2022: Computing continuous moments...
# [1] 0.6816645
```

## :snake: Ray-supported Usage

_This section is for users who (1) have successfully set up our software; and (2) want to explore ways to speed things up further on the Python backend. We do not recommend the steps below to those unfamiliar with Python (especially with installing libraries involving multiple dependencies and specific Python versions)._

For those familiar with Python and distributed computing, parallelization can help speed up _p_-value computations. This is especially useful when the CPU has many cores (20 or more). One way to set up parallelization is through [**ray**](https://docs.ray.io/en/latest/index.html). We outline the necessary steps to enable ray support for the back-end Python functions. 

1. (Ray installation) Install **ray**.

```
reticulate::py_install("ray", pip = TRUE)
``` 

2. (Activate R interface to **ray**-supported functions) In `main_draft0.R`, comment Line 8 and uncomment Line 9. The first few lines should look like this:

```
# Load functions in back-end R script 
source("local_functions.R")
# Load functions written in Python 
#reticulate::source_python("../python/local_functions_for_R.py")
reticulate::source_python("../python/local_functions.py")
``` 

Run `source("main_draft0.R")` after that. This ensures that R calls the **ray**-supported Python functions, rather than the default functions without **ray** support.

:snake: To test for successful implementation of the **ray**-supported backend, users may try running the following lines of code. 

```
# Use Python backend with ray support
x1 <- abs(rnorm(10)); y1 <- abs(rnorm(50))
mochis.test(x = x1,
            y = y1,
            p = 2, wList = rep(1,11),
            alternative = "two.sided",
            approx = "bernstein", n_mom = 50,
            python_backend = TRUE)
```

The curious user can also compare the computation against native R backend (with parallelization), and see for themselves the (tremendous) speedups enjoyed by **ray**'s parallelization.

```
# Use native R backend
registerDoParallel()
mochis.test(x = x1,
            y = y1,
            p = 2, wList = rep(1,11),
            alternative = "two.sided",
            approx = "bernstein", n_mom = 50,
            python_backend = FALSE)
```
 

Note: Some users might experience the following error after running the code chunk above.

```
ERROR services.py:1254 -- Failed to start the dashboard: Failed to read dashbord log: [Errno 2] No such file or directory: '/tmp/ray/session_2022-03-02_10-07-17_648103_26559/logs/dashboard.log'

(raylet)     import aiohttp.signals
(raylet) ModuleNotFoundError: No module named 'aiohttp.signals'
```

We found that this error is caused by earlier versions of **ray** (up to v1.5.2, at least) depending on an older version of the **aiohttp** package (see [this thread](https://github.com/ray-project/ray/issues/20681)). To fix this problem, one has to install the earlier **aiohttp** package. This can be done by pinning the version during installation. To do so using the terminal:

```
pip install aiohttp==3.7
```

To do so using R's **reticulate**:

```
reticulate::py_install("aiohttp==3.7")
``` 

## Application to scRNA-seq Data

We apply our methods to _Tabula Muris Senis_, a publicly available single cell RNA-seq dataset provided by the Chan-Zuckerberg Biohub. Details are provided in the `tabula-muris-analysis` subdirectory.
