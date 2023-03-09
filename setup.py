from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

external_deps = [
    'extern/eigen-3.4.0/',
    'extern/pcg-cpp-0.98/include/',
    'extern/pybind11/include/',
    '/usr/include/python3.11',
]

hybrid_rcc = Pybind11Extension(
    'hybrid_rcc',
    [str(fname) for fname in Path('stats/statistical_tests').rglob('*.cc')],
    include_dirs=external_deps + ["."],
    extra_compile_args=['-O3']
)

setup(
    name='hybrid_rcc',
    version=0.1,
    author='Noureldin Yosri',
    author_email='noureldinyosri@gmail.com',
    description='TBA',
    ext_modules=[hybrid_rcc],
    cmdclass={"build_ext": build_ext},
)

