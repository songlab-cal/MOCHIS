# Python User Manual

1. (Download required files) Download this directory. Also download the following zip files.
    - `chebyshev.zip` ([link](https://www.dropbox.com/s/t2ml80d3pl6p83q/chebyshev.zip?dl=0))
    - `jacobi.zip` ([link](https://www.dropbox.com/s/2envgy7q99ntkmj/jacobi.zip?dl=0))

2. Download anaconda and create a virtual environment.
```
conda create -n [NAME] python==3.9
conda activate [NAME]
```

3. Install package dependencies within the virtual environment
    - [**ray**](https://docs.ray.io/en/latest/installation.html)
    - [**combalg-py**]([https://numpy.org/](https://pythonhosted.org/combalg-py/))
    
```
pip install ray
pip install combalg-py
```

4. Install MOCHIS
```
[TODO]
pip install -i [TODO] MOCHIS==0.0.0
```

5. (Setup lookup table files) Create a directory called `chebyshev`, then unzip `chebyshev.zip` and put all folders (`m25`, `m50`, etc.) under `chebyshev`. Do the same for `jacobi`.  

    At this stage, you should have the following directory structure under your current working directory.

    - `mochis.py`
    - `auxiliary.py`
    - `chebyshev`
        - `m25`
        - `m50`
        - `...`
    - `jacobi` 
        - `m25`
        - `m50`
        - `...`

## Examples

We simulate two samples and then show how we can run our tests on them. We set the seed for our simulation to ensure reproducibility.

```
# Specific samples
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
```

```
mochis_py(x=x0, y=y0, p=2, wList=[1 for i in range(11)], alternative="two.sided", approx="chebyshev", n_mom=200)

# Normalizing weight vector...
# The test statistic for the data is  0.14619999999999997
# Sample size, n, is large enough, using Sk distribution...
# Computing continuous moments...
# 0.68166445583817
```
