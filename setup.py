from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension("solutions", ["solutions.pyx"]
              , language="c++"
              )
]
setup(
    name="solutions",
    ext_modules = cythonize(extensions, compiler_directives={'language_level' : "3"}),
)