from setuptools import setup

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='MOCHIS',
    version='1.2.1',
    description='Welcome to MOCHIS (MOment Computation and Hypothesis testing Induced by S_n,k), software for implementing flexible and exact two-sample tests, with applications to single cell genomics. It is based on the work of Erdmann-Pham et al. (2022+).',
    author='Xurui Chen',
    author_email='xuruichen@berkeley.edu',
    license='MIT',
    packages=['MOCHIS'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    zip_safe=False,
    install_requires=[
        'protobuf==3.20.1',
        'gmpy2',
        'numpy',
        'scipy'
    ]
)
