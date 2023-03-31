from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

hybrid_rcc_module = Pybind11Extension(
    'hybrid_rcc',
    [str(fname) for fname in Path('.').rglob('*.cc')],
    include_dirs=['.']
    + [str(f) for f in Path('third_party').glob('*') if f.is_dir()],
    extra_compile_args=['-O3'],
)

setup(
    name='hybrid_rcc',
    version=0.1,
    author='Noureldin Yosri',
    ext_modules=[hybrid_rcc_module],
    cmdclass={'build_ext': build_ext},
)
