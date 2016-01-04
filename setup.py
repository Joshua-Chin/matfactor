import os
import platform

from setuptools import setup, find_packages, Extension

setup(
    name='matfactor',
    version='1.0.0'
    description='A small library for factorizing matrices'
    author='Joshua Chin',
    install_requires=['cython']
    cmdclass={'build_ext':build_ext},
    ext_modules=[
        Extension(
            '_factorize',
            ['_factorize.pyx'],
            extra_compile_args=['-fopenmp'],
            extra_link_args=['-fopenmp']
        )
    ]
)
