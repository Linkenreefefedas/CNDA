"""
Setup script for CNDA Python bindings.

Build and install:
    pip install .

Build in development mode:
    pip install -e .

Build with verbose output:
    pip install . -v
"""

import os
import sys
import subprocess
from pathlib import Path

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        import pybind11
        
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        pybind11_dir = pybind11.get_cmake_dir()

        cmake_args = [
            f'-DCNDA_PYTHON_OUTPUT_DIR={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            '-DBUILD_TESTING=OFF',
            f'-Dpybind11_DIR={pybind11_dir}',
        ]

        # Support CNDA_BOUNDS_CHECK via environment variable
        if os.environ.get('CNDA_BOUNDS_CHECK', '').lower() in ('1', 'on', 'true', 'yes'):
            cmake_args.append('-DCNDA_BOUNDS_CHECK=ON')

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if sys.platform.startswith('win'):
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += [f'-DCMAKE_BUILD_TYPE={cfg}']
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = f'{env.get("CXXFLAGS", "")} -DVERSION_INFO=\\"{self.distribution.get_version()}\\"'
        
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


setup(
    name='cnda',
    version='0.1.0',
    author='CNDA Contributors',
    description='Contiguous N-Dimensional Array library with zero-copy NumPy interoperability',
    long_description='',
    ext_modules=[CMakeExtension('cnda')],
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False,
    python_requires='>=3.9',
    install_requires=['pybind11>=2.6.0'],
)
