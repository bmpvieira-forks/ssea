'''
Created on Oct 9, 2013

@author: mkiyer
'''
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

#from distutils.sysconfig import get_python_inc, get_python_lib
import numpy
numpy_inc = numpy.get_include()

ext_modules = [Extension('ssea.kernel', 
                         sources=['ssea/kernel.pyx', 'ssea/rng.c'], 
                         include_dirs=[numpy_inc],
                         libraries=['m']),
               Extension('ssea.cfisher', 
                         sources=['ssea/cfisher.pyx'], 
                         include_dirs=[numpy_inc])]

setup(name='SSEA',
      description='Sample Set Enrichment Analysis',
      url='http://ssea.googlecode.com',
      author='matthew iyer, yashar niknafs',
      author_email='matthew.iyer@gmail.com',
      requires=['numpy', 'jinja2'],
      ext_modules=cythonize(ext_modules),
      packages={'ssea'},
      package_data={'ssea.templates': ['detailedreport.html',
                                       'overview.html']},
      scripts=['ssea/ssea.py'])
