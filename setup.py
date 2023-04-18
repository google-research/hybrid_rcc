import os
import pathlib
import shutil
import subprocess
import zipfile

from setuptools import setup
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info

try:
  from pybind11.setup_helpers import Pybind11Extension
except ImportError:
  print('Installation requires pybind11')
  subprocess.check_call('pip install pybind11'.split())
  from pybind11.setup_helpers import Pybind11Extension

cpp_libraries = {
    'eigen': (
        'https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip'
    ),
    'pcg_random': 'https://www.pcg-random.org/downloads/pcg-cpp-0.98.zip',
}


def _install(cmd, args):
  os.mkdir('third_party')
  for library, url in cpp_libraries.items():
    subprocess.check_call(f'wget -O third_party/{library}.zip {url}'.split())
    with zipfile.ZipFile(f'third_party/{library}.zip', 'r') as f:
      f.extractall('third_party')
  cmd(*args)
  shutil.rmtree('third_party')


class InstallCommand(install):
  """Installation Command."""
  def run(self):
    _install(install.do_egg_install, [self])


class DevelopCommand(develop):
  """Installation Command."""

  def run(self):
    _install(develop.run, [self])


class EggInfoCommand(egg_info):
  """Installation Command."""

  def run(self):
    _install(egg_info.run, [self])


hybrid_rcc_module = Pybind11Extension(
    'hybrid_rcc',
    [str(fname) for fname in pathlib.Path('src').rglob('*.cc')],
    include_dirs=['src', 'third_party/eigen-3.4.0', 'third_party/pcg-cpp-0.98'],
    extra_compile_args=['-O3'],
)

setup(
    name='hybrid_rcc',
    version=0.1,
    author='Noureldin Yosri',
    ext_modules=[hybrid_rcc_module],
    cmdclass={
        'install': InstallCommand,
        'develop': DevelopCommand,
        'egg_info': EggInfoCommand,
    },
    install_requires=[
        'numpy',
        'scipy',
    ],
)
