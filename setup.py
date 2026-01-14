import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import setuptools


# finding cuda location

def find_cuda_home():
    """Locate the CUDA installation."""
    cuda_home = os.environ.get('CUDA_HOME')
    if cuda_home:
        return cuda_home
    
    try:
        import shutil
        nvcc_path = shutil.which('nvcc')
        if nvcc_path:
            return os.path.dirname(os.path.dirname(nvcc_path))
    except ImportError:
        pass
        
    for path in ['/usr/local/cuda', '/usr/local/cuda-11.3', '/usr/local/cuda-12.0']:
        if os.path.exists(path):
            return path
        
    return None

CUDA_HOME = find_cuda_home()

if CUDA_HOME is None:
    print("WARNING: CUDA_HOME not found. Build may fail.")
    print("Set CUDA_HOME environment variable or ensure nvcc is in PATH.")
    library_dirs = []
    include_dirs = []
else:
    print(f"Found CUDA_HOME: {CUDA_HOME}")
    library_dirs = [os.path.join(CUDA_HOME, 'lib64')]
    include_dirs = [os.path.join(CUDA_HOME, 'include')]

# build config

class get_numpy_include(object):
    def __str__(self):
        import numpy
        return numpy.get_include()

class get_pybind_include(object):
    def __str__(self):
        import pybind11
        return pybind11.get_include()

def customize_compiler_for_nvcc(self):
    default_compiler_so = self.compiler_so
    super_compile = self._compile

    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            self.set_executable('compiler_so', 'nvcc')
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['cxx']
        super_compile(obj, src, ext, cc_args, postargs, pp_opts)
        self.compiler_so = default_compiler_so

    self._compile = _compile

class custom_build_ext(build_ext):
    def build_extensions(self):
        self.compiler.src_extensions.append('.cu')
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)

## adjust according to gpu ##
#   sm_60 - Pascal (GTX 1000 series)
#   sm_70 - Volta (V100)
#   sm_75 - Turing (RTX 2000 series)
#   sm_86 - Ampere (RTX 3000 series consumer)
#   sm_89 - Ada Lovelace (RTX 4000 series)
#   sm_90 - Hopper (H100)

cuda_arch = os.environ.get('CUDA_ARCH', 'sm_86')

ext_modules = [
    Extension(
        'convexiou_gpu',
        ['pybind_wrapper.cpp', 'convexiou_cuda.cu'],
        include_dirs=[
            get_pybind_include(),
            get_numpy_include(),
            '.',
        ] + include_dirs,
        libraries=['cudart'],
        library_dirs=library_dirs,
        runtime_library_dirs=library_dirs if library_dirs else None,
        language='c++',
        extra_compile_args={
            'cxx': ['-std=c++14', '-fPIC', '-O3'],
            'nvcc': [f'-arch={cuda_arch}', '--compiler-options', '-fPIC', '-c', '-O3']
        },
    ),
]

setup(
    name='convexiou_gpu',
    version='1.0.0',
    author='Gabriel',
    description='GPU-Accelerated Convex Polygon IoU for Ellipse Approximation',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.5.0', 'numpy'],
    install_requires=['numpy'],
    cmdclass={'build_ext': custom_build_ext},
    zip_safe=False,
    python_requires='>=3.7',
)
