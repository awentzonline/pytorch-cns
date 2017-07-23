#!/usr/bin/env python
from distutils.core import setup
from setuptools import find_packages

setup(
    name='pytorch-evonet',
    version='0.0.1',
    description='Evolve neural networks with PyTorch',
    author='Adam Wentz',
    author_email='adam@adamwentz.com',
    url='https://github.com/awentzonline/pytorch-evonet',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'torchvision',
        'scipy',
        'six',
    ]
)
