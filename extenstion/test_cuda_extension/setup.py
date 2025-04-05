from setuptools import setup, find_packages, Extension, Command
import glob
import os
import pybind11
import shutil

import torch
from torch.utils.cpp_extension import (
    CUDA_HOME,
    IS_WINDOWS,
    ROCM_HOME,
    SYCL_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
    SyclExtension,
    _get_cuda_arch_flags,
)


def get_extensions():
    # PY3_9_HEXCODE = "0x03090000"
    debug_mode = False
    extra_link_args = []

    extra_compile_args = {
        # "cxx": [f"-DPy_LIMITED_API={PY3_9_HEXCODE}", "-O3", "-fdiagnostics-color=always", '-g', '-std=c++20', '-fPIC'],
        "cxx": ["-O3", "-fdiagnostics-color=always", '-g', '-std=c++17', '-fPIC'],
        "nvcc": [
            "-DNDEBUG" if not debug_mode else "-DDEBUG",
            "-O3" if not debug_mode else "-O0",
            "-t=0",
            "-std=c++17",
        ],
    }
    # extra_compile_args = ["-std=c++11"]

    use_cuda = torch.cuda.is_available() and (
        CUDA_HOME is not None or ROCM_HOME is not None
    )
    extension = CUDAExtension if use_cuda else CppExtension
    assert SYCL_HOME is None, "this is a cuda extension"
    if SYCL_HOME:
        extension = SyclExtension

    extensions_dir = "torch_cuda_extension/csrc/"
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp"), recursive=True))
    include_dirs = []
    if CUDA_HOME:
        # sources = list(glob.glob(os.path.join(extensions_dir, "**/**/*.cu"), recursive=True))
        # include_dirs += ["/4T-720/leslie/inductor/pytorch/third_party/torch-xpu-ops/src/",]
        pass
    if SYCL_HOME:
        sources = list(glob.glob(os.path.join(extensions_dir, "**/*.cpp"), recursive=True))
        sources += list(glob.glob(os.path.join(extensions_dir, "**/*.sycl"), recursive=True))
        include_dirs += ["/4T-720/leslie/inductor/pytorch/third_party/torch-xpu-ops/src/",]

    ext = extension(
        "torch_cuda_extension._C",  # 生成的 Python 模块名
        sources,  # C++ 源文件路径
        py_limited_api=False,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,  # 额外的编译参数
        extra_link_args=extra_link_args,
    )
    ext_modules = [
        ext
    ]

    return ext_modules

class CustomCleanCommand(Command):
    """Custom clean command to remove additional files/directories."""
    
    # Specify the name of the command
    description = 'Custom clean command to remove build artifacts.'
    user_options = []
    
    def initialize_options(self):
        # No options for this example
        pass

    def finalize_options(self):
        # No options for this example
        pass

    def run(self):
        # Standard clean tasks (removes build directories)
        build_dirs = ['build', 'dist', 'torch_cuda_extension.egg-info']
        for build_dir in build_dirs:
            if os.path.exists(build_dir):
                print(f"Removing {build_dir}...", flush=True)
                shutil.rmtree(build_dir)
        
        # You can add custom clean-up logic here
        temp_files = list(glob.glob('torch_cuda_extension/_C*.so', recursive=True))
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                print(f"Removing {temp_file}...", flush=True)
                os.remove(temp_file)

        print("Custom clean process complete!")

setup(
    name="torch_cuda_extension",  # Project name
    version="0.1.0",  # Initial version
    author="Leslie Fang",
    packages=find_packages(),  # Automatically find all packages
    ext_modules=get_extensions(),
    cmdclass={
        'clean': CustomCleanCommand,  # Register our custom clean command
        'build_ext': BuildExtension,
    },
)