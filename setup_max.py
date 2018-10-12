'''
This is the central cython setup file, which compiles all the cython .pyx files of the project.
It has to be run whenever a change has been made to the files.
Targeted files:
- nilmtk/disaggregate/accelerators.pyx
- nilmtk/stats/accelerators_stat.pyx
'''

#  python setup.py build_ext --inplace

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize(["nilmtk/disaggregate/accelerators.pyx", "nilmtk/stats/accelerators_stat.pyx"], language="c++"), 
    include_dirs=[numpy.get_include()]
)




##Use this setup.py if you want setup to automatically cythonize all pyx in the codeRootFolder
##To run this setup do exefile('pathToThisSetup.py')

#import os
#from distutils.core import setup
#from distutils.extension import Extension
#from Cython.Distutils import build_ext

#def scandir(dir, files=[]):
#    for file in os.listdir(dir):
#        path = os.path.join(dir, file)
#        if os.path.isfile(path) and path.endswith(".pyx"):
#            files.append(path.replace(os.path.sep, ".")[:-4])
#        elif os.path.isdir(path):
#            scandir(path, files)
#    return files


#def makeExtension(extName):
#    extPath = extName.replace(".", os.path.sep)+".pyx"
#    return Extension(
#        extName,
#        [extPath],
#        include_dirs = ['.'] #your include_dirs must contains the '.' for setup to search all the subfolder of the codeRootFolder
#        )


#extNames = scandir('codeRootFolder')

#extensions = [makeExtension(name) for name in extNames]

#setup(
#  name="workingCythonMultiPackageProject",
#  ext_modules=extensions,
#  cmdclass = {'build_ext': build_ext},
#  script_args = ['build_ext'],
#  options = {'build_ext':{'inplace':True, 'force':True}}
#)

#print '********CYTHON COMPLETE******'