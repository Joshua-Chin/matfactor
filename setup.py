from setuptools import setup, find_packages, Extension

try:
    from Cython.Distutils import build_ext
except:
    print("You don't seem to have cython installed.")
    print("Please install it by running sudo pip install cython")
    import sys
    sys.exit(1)

setup(
    name='matfactor',
    version='1.0.0',
    description='A small library for factorizing matrices',
    author='Joshua Chin',
    install_requires=['cython'],
    cmdclass={'build_ext':build_ext},
    ext_modules=[
        Extension(
            'matfactor._factorize',
            ['matfactor/_factorize.pyx'],
            extra_compile_args=['-fopenmp'],
            extra_link_args=['-fopenmp']
        )
    ]
)
