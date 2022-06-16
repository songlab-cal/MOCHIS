"""
Unit tests for MOCHIS

@author: Alan Aw and Rachel Xurui Chen
"""
from MOCHIS.auxiliary import get_composition_pvalue
from MOCHIS.auxiliary import _compositions
from MOCHIS.auxiliary import _single_comp
from MOCHIS.mochis import mochis_py
import numpy as np


def test_single_comp():
    cert_arr = []
    for i in range(100):
        n = np.random.random_integers(low=30,high=100)
        k = np.random.random_integers(low=5,high=20)
        one_comp = _single_comp(n=n,k=k)
        cert_arr.append(np.sum(one_comp)==n and len(one_comp)==k)
    assert cert_arr.all(), '_single_comp failed test'


def test_compositions():
    cert_arr = []
    vec_len=np.vectorize(len)
    vec_sum=np.vectorize(sum)
    for i in range(100):
        n = np.random.random_integers(low=30,high=100)
        k = np.random.random_integers(low=5,high=20)
        nsamp = np.random.random_integers(low=20,high=30)
        list_of_comps = _compositions(n=n,k=k,nsamples=nsamp)
        cert_arr.append(vec_len(list_of_comps)==[k for i in range(nsamp)] and \
            vec_sum(list_of_comps)==[n for i in range(nsamp)] and \
            len(list_of_comps) == nsamp)
    assert cert_arr.all(), '_compositions failed test'

def test_p1_composition():
    p = 1
    wList = [np.random.uniform(low=1,high=10) for i in range(11)]
    p_val_vec = []
    for i in range(1000):
        x = np.random.default_rng().normal(0, 1, 10)
        y = np.random.default_rng().normal(0, 1, 30)
        x_ordered = np.sort(np.array(x))
        x_ordered = np.insert(x_ordered, 0, -math.inf)
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
    
    assert np.allclose([np.mean(p_val_vec),np.var(p_val_vec)], [0.5,1/12]),  'Distribution of'
    ' p-values (p=1) is not uniform'
    assert np.min(p_val_vec) > 0,  'Minimum p-value (p=1) is not positive'


def test_p2_composition():
    p = 2
    wList = [np.random.uniform(low=1,high=10) for i in range(11)]
    p_val_vec = []
    for i in range(1000):
        x = np.random.default_rng().normal(0, 1, 10)
        y = np.random.default_rng().normal(0, 1, 30)
        x_ordered = np.sort(np.array(x))
        x_ordered = np.insert(x_ordered, 0, -math.inf)
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
    
    assert np.allclose([np.mean(p_val_vec),np.var(p_val_vec)], [0.5,1/12]),  'Distribution of'
    ' p-values (p=2) is not uniform'
    assert np.min(p_val_vec) > 0,  'Minimum p-value (p=2) is not positive'


def test_p1_gaussian():
    p = 1
    wList = [np.random.uniform(low=1,high=10) for i in range(51)]
    p_val_vec = []
    for i in range(1000):
        x = np.random.default_rng().normal(0, 1, 50)
        y = np.random.default_rng().normal(0, 1, 100)
        p_val = mochis_py(x=x, y=y, p=p, wList=wList)
        p_val_vec.append(p_val)
    
    assert np.allclose([np.mean(p_val_vec),np.var(p_val_vec)], [0.5,1/12]),  'Distribution of'
    ' p-values (p=1) is not uniform'


def test_p2_gaussian():
    p = 2
    wList = [np.random.uniform(low=1,high=10) for i in range(51)]
    p_val_vec = []
    for i in range(1000):
        x = np.random.default_rng().normal(0, 1, 50)
        y = np.random.default_rng().normal(0, 1, 100)
        p_val = mochis_py(x=x, y=y, p=p, wList=wList)
        p_val_vec.append(p_val)
    
    assert np.allclose([np.mean(p_val_vec),np.var(p_val_vec)], [0.5,1/12]),  'Distribution of'
    ' p-values (p=2) is not uniform'