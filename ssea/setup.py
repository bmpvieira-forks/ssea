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

ext_modules = [Extension('ssea.lib.kernel', 
                         sources=['ssea/lib/kernel.pyx', 'ssea/lib/rng.c'], 
                         include_dirs=[numpy_inc],
                         libraries=['m']),
               Extension('ssea.lib.cfisher', 
                         sources=['ssea/lib/cfisher.pyx'], 
                         include_dirs=[numpy_inc])]

setup(name='SSEA',
      description='Sample Set Enrichment Analysis',
      url='http://ssea.googlecode.com',
      author='matthew iyer, yashar niknafs',
      author_email='matthew.iyer@gmail.com',
      requires=['numpy', 'jinja2'],
      ext_modules=cythonize(ext_modules),
      packages={'ssea'},
      package_data={'ssea.templates': ['details.html',
                                       'report.html']},
      scripts=['ssea/ssea', 
               'ssea/report'])
