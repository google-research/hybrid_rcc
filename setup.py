import os
import pathlib
import shutil
import subprocess
import zipfile

from setuptools import setup, Extension
from setuptools.command import build_ext

cpp_libraries = {
    'eigen': (
        'https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip'
    ),
    'pcg_random': 'https://www.pcg-random.org/downloads/pcg-cpp-0.98.zip',
}

_THIRD_PARTY = '_third_party_'


class BuildExtCommand(build_ext.build_ext):
  def initialize_options(self):
      if not os.path.exists(_THIRD_PARTY):
        os.mkdir(_THIRD_PARTY)
        for library, url in cpp_libraries.items():
          subprocess.check_call(
              f'wget -O {_THIRD_PARTY}/{library}.zip {url}'.split()
          )
          with zipfile.ZipFile(f'{_THIRD_PARTY}/{library}.zip', 'r') as f:
            f.extractall(_THIRD_PARTY)
      super().initialize_options()

  def finalize_options(self):
      from pybind11.setup_helpers import Pybind11Extension
      self.distribution.ext_modules[:] = [
        Pybind11Extension(
          'hybrid_rcc',
          [str(fname) for fname in pathlib.Path('src').rglob('*.cc')],
          include_dirs=[
              'src',
              f'{_THIRD_PARTY}/eigen-3.4.0',
              f'{_THIRD_PARTY}/pcg-cpp-0.98',
          ],
          extra_compile_args=['-O3'],
        )
      ]
      super().finalize_options()

  def build_extensions(self):
    super().build_extensions()  
    shutil.rmtree(_THIRD_PARTY)

    
setup(
    name='hybrid_rcc',
    version=0.1,
    author='Noureldin Yosri',
    ext_modules=[Extension('.', [])],
#     packages=['hybrid_rcc'],
    package_data={'.': ['src/py/hybrid_rcc.pyi']},
    cmdclass={
        'build_ext': BuildExtCommand,
    },
    setup_requires=[
        'numpy',
        'pybind11',        
    ],
    install_requires=[
        'numpy',
        'scipy',
    ],
)
