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
    long_description="A small library for factorizing matrices. Given a sparse matrix M, it finds matrices u,v that minimize M - e ** (u*v)",
    author='Joshua Chin',
    author_email='JoshuaRChin@gmail.com',
    url='https://github.com/Joshua-Chin/matfactor/tree/master',
    license='License :: OSI Approved :: Apache Software License',
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
