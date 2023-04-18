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

_THIRD_PARTY = '_third_party_'


def _install(cmd, args):
  if not os.path.exists(_THIRD_PARTY):
    os.mkdir(_THIRD_PARTY)
    for library, url in cpp_libraries.items():
      subprocess.check_call(
          f'wget -O {_THIRD_PARTY}/{library}.zip {url}'.split()
      )
      with zipfile.ZipFile(f'{_THIRD_PARTY}/{library}.zip', 'r') as f:
        f.extractall(_THIRD_PARTY)
  print(os.listdir(_THIRD_PARTY))
  cmd(*args)
  shutil.rmtree(_THIRD_PARTY)


class InstallCommand(install):
  """Installation Command."""
  def run(self):
    _install(install.do_egg_install, [self])


class DevelopCommand(develop):
  """Installation Command."""

  def run(self):
    _install(develop.run, [self])


class EggInfoCommand(egg_info):
  """EggInfo Command."""

  def run(self):
    self.run_command('install')
    egg_info.run(self)


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
        'install': InstallCommand,
        'egg_info': EggInfoCommand,
    },
    install_requires=[
        'numpy',
        'scipy',
    ],
)
