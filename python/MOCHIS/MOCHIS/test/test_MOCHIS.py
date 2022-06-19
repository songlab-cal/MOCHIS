"""
Unit tests for MOCHIS

@author: Alan Aw and Rachel Xurui Chen
"""


import sys
sys.path.append('..')


from auxiliary import get_composition_pvalue
from auxiliary import _compositions
from auxiliary import _single_comp
from mochis import mochis_py
import numpy as np

import math

import os

#Source: https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print#:~:text=If%20you%20don't%20want,the%20top%20of%20the%20file.    
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def test_single_comp():
    cert_arr = []
    for i in range(100):
        n = np.random.random_integers(low=30,high=100)
        k = np.random.random_integers(low=5,high=20)
        one_comp = _single_comp(n=n,k=k)
        cert_arr.append(np.sum(one_comp)==n and len(one_comp)==k)
    assert np.array(cert_arr).all(), '_single_comp failed test'
    print("test_single_comp PASS")


def test_compositions():
    cert_arr = []
    #vec_len=np.vectorize(len)
    #vec_sum=np.vectorize(sum)
    for i in range(100):
        n = np.random.random_integers(low=30,high=100)
        k = np.random.random_integers(low=5,high=20)
        nsamp = np.random.random_integers(low=20,high=30)
        list_of_comps = _compositions(n=n,k=k,nsample=nsamp)
        cert_arr.append([len(i) for i in list_of_comps]==[k for i in range(nsamp)] and \
            [sum(i) for i in list_of_comps]==[n for i in range(nsamp)] and \
            len(list_of_comps) == nsamp)
    assert np.array(cert_arr).all(), '_compositions failed test'
    print("test_compositions PASS")

def test_p1_composition():
    p = 1
    wList = [np.random.uniform(low=1,high=10) for i in range(11)]
    p_val_vec = []
    for i in range(1000):
        x = np.random.default_rng().normal(0, 1, 10)
        y = np.random.default_rng().normal(0, 1, 30)
        x_ordered = np.sort(np.array(x))
        x_ordered = np.insert(x_ordered, 0, -math.inf)
        x_ordered = np.append(x_ordered, math.inf)
        k = len(x) + 1
        n = len(y)
        
        # Construct Snk
        snk = []
        for i in range(k):
            snk.append(((x_ordered[i] <= y) & (y < x_ordered[i+1])).sum())
        
        # Construct t
        t_arr = [((snk[i]/n)**p) * wList[i] for i in range(k)]
        t = sum(t_arr)
        
        p_val = get_composition_pvalue(t=t,n=n,k=k,p=p,wList=wList,alternative="two.sided", resamp_number=5000,type="valid")
        p_val_vec.append(p_val)
    
    
    assert abs(np.mean(p_val_vec) - 0.5) <= 0.05, "Mean deviates too much - distribution of p-values (p=1) is not uniform"
    assert abs(np.var(p_val_vec) - 1/12) <= 0.01, "Variance deviates too much - distribution of p-values (p=1) is not uniform"

    assert np.min(p_val_vec) > 0,  'Minimum p-value (p=1) is not positive'
    print("test_p1_composition PASS")


def test_p2_composition():
    p = 2
    wList = [np.random.uniform(low=1,high=10) for i in range(11)]
    p_val_vec = []
    for i in range(1000):
        x = np.random.default_rng().normal(0, 1, 10)
        y = np.random.default_rng().normal(0, 1, 30)
        x_ordered = np.sort(np.array(x))
        x_ordered = np.insert(x_ordered, 0, -math.inf)
        x_ordered = np.append(x_ordered, math.inf)
        k = len(x) + 1
        n = len(y)
        
        # Construct Snk
        snk = []
        for i in range(k):
            snk.append(((x_ordered[i] <= y) & (y < x_ordered[i+1])).sum())
        
        # Construct t
        t_arr = [((snk[i]/n)**p) * wList[i] for i in range(k)]
        t = sum(t_arr)
        
        p_val = get_composition_pvalue(t=t,n=n,k=k,p=p,wList=wList,alternative="two.sided", resamp_number=5000,type="valid")
        p_val_vec.append(p_val)
    
    assert abs(np.mean(p_val_vec) - 0.5) <= 0.05, "Mean deviates too much - distribution of p-values (p=1) is not uniform"
    assert abs(np.var(p_val_vec) - 1/12) <= 0.01, "Variance deviates too much - distribution of p-values (p=1) is not uniform"

    assert np.min(p_val_vec) > 0,  'Minimum p-value (p=2) is not positive'
    print("test_p2_composition PASS")


def test_p1_gaussian():
    p = 1
    wList = [np.random.uniform(low=1,high=10) for i in range(51)]
    p_val_vec = []
    for i in range(1000):
        x = np.random.default_rng().normal(0, 1, 50)
        y = np.random.default_rng().normal(0, 1, 100)
        with HiddenPrints():
            p_val = mochis_py(x=x, y=y, p=p, wList=wList)
        p_val_vec.append(p_val)
    
    assert abs(np.mean(p_val_vec) - 0.5) <= 0.05, "Mean deviates too much - distribution of p-values (p=1) is not uniform"+str(np.mean(p_val_vec))
    assert abs(np.var(p_val_vec) - 1/12) <= 0.01, "Variance deviates too much - distribution of p-values (p=1) is not uniform"+str(np.var(p_val_vec))
    print("test_p1_gaussian PASS")



def test_p2_gaussian():
    p = 2
    wList = [np.random.uniform(low=1,high=10) for i in range(51)]
    p_val_vec = []
    for i in range(1000):
        x = np.random.default_rng().normal(0, 1, 50)
        y = np.random.default_rng().normal(0, 1, 100)
        
        with HiddenPrints():
            p_val = mochis_py(x=x, y=y, p=p, wList=wList)
        p_val_vec.append(p_val)
    
    assert abs(np.mean(p_val_vec) - 0.5) <= 0.05, "Mean deviates too much - distribution of p-values (p=2) is not uniform"+str(np.mean(p_val_vec))
    assert abs(np.var(p_val_vec) - 1/12) <= 0.01, "Variance deviates too much - distribution of p-values (p=2) is not uniform"+str(np.var(p_val_vec))
    print("test_p2_gaussian PASS")



if __name__ == "__main__":
    test_single_comp()
    test_compositions()
    test_p1_composition()
    test_p2_composition()
    test_p1_gaussian()
    test_p2_gaussian()