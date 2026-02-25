import os
import sys
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import setuptools

IS_WINDOWS = sys.platform == 'win32'


def find_cuda_home():
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home:
        return cuda_home

    try:
        import shutil
        nvcc_path = shutil.which('nvcc')
        if nvcc_path:
            return os.path.dirname(os.path.dirname(nvcc_path))
    except ImportError:
        pass

    if IS_WINDOWS:
        base = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
        if os.path.isdir(base):
            versions = sorted(os.listdir(base), reverse=True)
            for v in versions:
                candidate = os.path.join(base, v)
                if os.path.exists(os.path.join(candidate, 'bin', 'nvcc.exe')):
                    return candidate
    else:
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
    if IS_WINDOWS:
        library_dirs = [os.path.join(CUDA_HOME, 'lib', 'x64')]
    else:
        library_dirs = [os.path.join(CUDA_HOME, 'lib64')]
    include_dirs = [os.path.join(CUDA_HOME, 'include')]


class get_numpy_include(object):
    def __str__(self):
        import numpy
        return numpy.get_include()


class get_pybind_include(object):
    def __str__(self):
        import pybind11
        return pybind11.get_include()


## adjust according to gpu ##
#   sm_60 - Pascal (GTX 1000 series)
#   sm_70 - Volta (V100)
#   sm_75 - Turing (RTX 2000 series)
#   sm_86 - Ampere (RTX 3000 series consumer)
#   sm_89 - Ada Lovelace (RTX 4000 series)
#   sm_90 - Hopper (H100)

cuda_arch = os.environ.get('CUDA_ARCH', 'sm_75')


class custom_build_ext(build_ext):
    def build_extensions(self):
        self.compiler.src_extensions.append('.cu')

        original_compile = self.compiler.compile
        original_link = self.compiler.link

        nvcc_bin = 'nvcc'
        if CUDA_HOME:
            candidate = os.path.join(CUDA_HOME, 'bin', 'nvcc.exe' if IS_WINDOWS else 'nvcc')
            if os.path.exists(candidate):
                nvcc_bin = candidate

        cl_dir = None
        if IS_WINDOWS:
            import shutil
            cl_path = shutil.which('cl')
            if cl_path:
                cl_dir = os.path.dirname(cl_path)
            else:
                try:
                    cc = self.compiler.cc
                    if cc and os.path.isfile(cc):
                        cl_dir = os.path.dirname(cc)
                except AttributeError:
                    pass
                if not cl_dir:
                    try:
                        cc = self.compiler.compiler_cxx[0] if hasattr(self.compiler, 'compiler_cxx') else None
                        if cc and os.path.isfile(cc):
                            cl_dir = os.path.dirname(cc)
                    except (AttributeError, IndexError):
                        pass
                if not cl_dir:
                    import glob
                    for pattern in [
                        r"C:\Program Files (x86)\Microsoft Visual Studio\*\BuildTools\VC\Tools\MSVC\*\bin\HostX86\x64\cl.exe",
                        r"C:\Program Files (x86)\Microsoft Visual Studio\*\BuildTools\VC\Tools\MSVC\*\bin\HostX64\x64\cl.exe",
                        r"C:\Program Files\Microsoft Visual Studio\*\BuildTools\VC\Tools\MSVC\*\bin\HostX64\x64\cl.exe",
                        r"C:\Program Files (x86)\Microsoft Visual Studio\*\Community\VC\Tools\MSVC\*\bin\HostX64\x64\cl.exe",
                        r"C:\Program Files\Microsoft Visual Studio\*\Community\VC\Tools\MSVC\*\bin\HostX64\x64\cl.exe",
                    ]:
                        matches = sorted(glob.glob(pattern), reverse=True)
                        if matches:
                            cl_dir = os.path.dirname(matches[0])
                            break

        def custom_compile(sources, output_dir=None, macros=None, include_dirs=None,
                           debug=0, extra_preargs=None, extra_postargs=None, depends=None):
            cu_sources = [s for s in sources if os.path.splitext(s)[1] == '.cu']
            cpp_sources = [s for s in sources if os.path.splitext(s)[1] != '.cu']

            objects = []

            if cpp_sources:
                if isinstance(extra_postargs, dict):
                    cpp_postargs = extra_postargs.get('cxx', [])
                else:
                    cpp_postargs = extra_postargs or []
                objects += original_compile(cpp_sources, output_dir, macros,
                                            include_dirs, debug, extra_preargs,
                                            cpp_postargs, depends)

            for cu_src in cu_sources:
                base = os.path.splitext(os.path.basename(cu_src))[0]
                obj_ext = '.obj' if IS_WINDOWS else '.o'
                obj_file = os.path.join(output_dir or '.', base + obj_ext)
                os.makedirs(os.path.dirname(obj_file) or '.', exist_ok=True)

                nvcc_cmd = [nvcc_bin, '-c', cu_src, '-o', obj_file,
                            f'-arch={cuda_arch}', '-O3',
                            '--allow-unsupported-compiler']

                if CUDA_HOME:
                    nvcc_cmd += ['-I', os.path.join(CUDA_HOME, 'include')]

                if IS_WINDOWS:
                    nvcc_cmd += ['--compiler-options', '/O2,/EHsc,/MD']
                else:
                    nvcc_cmd += ['--compiler-options', '-fPIC']

                if include_dirs:
                    for d in include_dirs:
                        nvcc_cmd += ['-I', str(d)]

                if macros:
                    for macro in macros:
                        if len(macro) == 2 and macro[1] is not None:
                            nvcc_cmd += [f'-D{macro[0]}={macro[1]}']
                        else:
                            nvcc_cmd += [f'-D{macro[0]}']

                print(f"nvcc: {' '.join(nvcc_cmd)}")
                env = os.environ.copy()
                if IS_WINDOWS and cl_dir:
                    env['PATH'] = cl_dir + ';' + env.get('PATH', '')
                subprocess.check_call(nvcc_cmd, env=env)
                objects.append(obj_file)

            return objects

        self.compiler.compile = custom_compile
        build_ext.build_extensions(self)


if IS_WINDOWS:
    cxx_args = ['/O2', '/EHsc', '/std:c++14', '/wd4005']
else:
    cxx_args = ['-std=c++14', '-fPIC', '-O3']

ext_modules = [
    Extension(
        'convexiou._core',
        ['pybind_wrapper.cpp', 'convexiou_cuda.cu'],
        include_dirs=[
            get_pybind_include(),
            get_numpy_include(),
            '.',
        ],
        libraries=['cudart'],
        library_dirs=library_dirs,
        runtime_library_dirs=library_dirs if (library_dirs and not IS_WINDOWS) else None,
        language='c++',
        extra_compile_args={'cxx': cxx_args, 'nvcc': []},
    ),
]

setup(
    name='convexiou',
    version='2.0.0',
    author='Gabriel',
    description='GPU-accelerated IoU for oriented bounding boxes via ellipse/polygon approximation',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    packages=['convexiou'],
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.5.0', 'numpy'],
    install_requires=['numpy'],
    cmdclass={'build_ext': custom_build_ext},
    zip_safe=False,
    python_requires='>=3.8',
)
