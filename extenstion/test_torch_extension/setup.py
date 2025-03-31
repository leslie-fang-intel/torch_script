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
    BuildExtension,
    CppExtension,
    CUDAExtension,
    _get_cuda_arch_flags,
)


def get_extensions():
    PY3_9_HEXCODE = "0x03090000"
    debug_mode = False
    extra_link_args = []

    extra_compile_args = {
        "cxx": [f"-DPy_LIMITED_API={PY3_9_HEXCODE}", "-O3", "-fdiagnostics-color=always"],
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

    extensions_dir = "test_ll_extension/csrc/"
    sources = list(glob.glob(os.path.join(extensions_dir, "**/*.cpp"), recursive=True))

    ext = extension(
        "test_ll_extension._C",  # 生成的 Python 模块名
        sources,  # C++ 源文件路径
        py_limited_api=True,
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
        build_dirs = ['build', 'dist', 'test_ll_extension.egg-info']
        for build_dir in build_dirs:
            if os.path.exists(build_dir):
                print(f"Removing {build_dir}...")
                shutil.rmtree(build_dir)
        
        # You can add custom clean-up logic here
        temp_files = ['test_ll_extension/example.cpython-310-x86_64-linux-gnu.so', 'test_ll_extension/_C.abi3.so']  # Example temporary files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                print(f"Removing {temp_file}...")
                os.remove(temp_file)

        print("Custom clean process complete!")

setup(
    name="test_ll_extension",  # Project name
    version="0.1.0",  # Initial version
    author="Leslie Fang",
    packages=find_packages(),  # Automatically find all packages
    ext_modules=get_extensions(),
    cmdclass={
        'clean': CustomCleanCommand,  # Register our custom clean command
        'build_ext': BuildExtension,
    },
    # install_requires=[
    #     "numpy>=1.21.0",
    #     "requests",
    # ],  # Dependencies from PyPI
    # classifiers=[
    #     "Programming Language :: Python :: 3",
    #     "License :: OSI Approved :: MIT License",
    #     "Operating System :: OS Independent",
    # ],
    # python_requires=">=3.7",
    # entry_points={
    #     "console_scripts": [
    #         "my_project=my_project.module1:main",  # CLI command
    #     ],
    # },
)