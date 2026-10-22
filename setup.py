"""Opt-in compiled DESIGNER RPG backend; default distribution stays Python-only."""
import os
from setuptools import Extension, setup

extensions = []
if os.environ.get("PYDKE_BUILD_RPG") == "1":
    import pybind11
    prefix = os.environ.get("FFTW_PREFIX", "/usr/local")
    extensions = [Extension(
        "pydesigner.preprocessing._rpg",
        ["pydesigner/preprocessing/rpg_backend/unring_rpg_pybind.cpp"],
        include_dirs=[pybind11.get_include(), prefix + "/include"],
        library_dirs=[prefix + "/lib"],
        libraries=["fftw3", "fftw3_threads"],
        extra_compile_args=["-std=c++11", "-pthread"],
        extra_link_args=["-pthread"],
        language="c++",
    )]
setup(ext_modules=extensions)
