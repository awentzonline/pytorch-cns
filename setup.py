#!/usr/bin/env python
from distutils.core import setup
from setuptools import find_packages

setup(
    name='pytorch-cns',
    version='0.2.2',
    description='Generalized Compressed Network Search with PyTorch',
    author='Adam Wentz',
    author_email='adam@adamwentz.com',
    url='https://github.com/awentzonline/pytorch-cns',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'torchvision',
        'scipy',
        'six',
        'redis',
    ]
)
