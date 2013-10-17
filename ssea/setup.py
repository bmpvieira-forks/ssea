'''
Created on Oct 9, 2013

@author: mkiyer
'''
import os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

#from distutils.sysconfig import get_python_inc, get_python_lib
import numpy
numpy_inc = numpy.get_include()

ext_modules = [Extension('src.kernel', ["src/kernel.pyx"], include_dirs=[numpy_inc])] 

setup(
  name = 'SSEA',
  ext_modules = cythonize(ext_modules)
)
