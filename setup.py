from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension("solutions", ["solutions.pyx"],
              extra_compile_args=['-openmp', '-O2'], extra_link_args=['-openmp'])
]
setup(
    name="hsshipper",
    ext_modules = cythonize(extensions, compiler_directives={'language_level' : "3"}),
)