import os
import pathlib
import shutil
import subprocess
import zipfile

from setuptools import setup
import setuptools.command.build_ext

try:
  from pybind11.setup_helpers import Pybind11Extension, build_ext
except ImportError:
  print('Installation requires pybind11')
  subprocess.check_call('pip install pybind11'.split())
  from pybind11.setup_helpers import Pybind11Extension, build_ext

cpp_libraries = {
    'eigen': (
        'https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip'
    ),
    'pcg_random': 'https://www.pcg-random.org/downloads/pcg-cpp-0.98.zip',
}

_THIRD_PARTY = '_third_party_'


class BuildExtCommand(build_ext):
  def initialize_options(self):
      if not os.path.exists(_THIRD_PARTY):
        os.mkdir(_THIRD_PARTY)
        for library, url in cpp_libraries.items():
          subprocess.check_call(
              f'wget -O {_THIRD_PARTY}/{library}.zip {url}'.split()
          )
          with zipfile.ZipFile(f'{_THIRD_PARTY}/{library}.zip', 'r') as f:
            f.extractall(_THIRD_PARTY)

  def finalize_options(self):
    shutil.rmtree(_THIRD_PARTY)
    super().finalize_options()  

hybrid_rcc_module = Pybind11Extension(
    'hybrid_rcc',
    [str(fname) for fname in pathlib.Path('src').rglob('*.cc')],
    include_dirs=[
        'src',
        f'{_THIRD_PARTY}/eigen-3.4.0',
        f'{_THIRD_PARTY}/pcg-cpp-0.98',
    ],
    extra_compile_args=['-O3'],
)

setup(
    name='hybrid_rcc',
    version=0.1,
    author='Noureldin Yosri',
    ext_modules=[hybrid_rcc_module],
    cmdclass={
        'build_ext': BuildExtCommand,
    },
    install_requires=[
        'numpy',
        'scipy',
    ],
)
