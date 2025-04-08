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
from pathlib import Path

package_name = "test_ll_extension"

def get_extensions():
    PY3_9_HEXCODE = "0x03090000"
    debug_mode = False
    use_cutlass = False
    my_file = Path(os.path.join(Path(f"{package_name}").resolve(), "include/cutlass/include/"))
    if my_file.is_dir():
        use_cutlass = True
    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            f"-DPy_LIMITED_API={PY3_9_HEXCODE}",
            "-O3",
            "-fdiagnostics-color=always",
            '-g',
            '-std=c++20',
            '-fPIC',
        ],
        "nvcc": [
            "-DNDEBUG" if not debug_mode else "-DDEBUG",
            "-O3" if not debug_mode else "-O0",
            "-t=0",
            "-std=c++17",
        ],
    }

    if use_cutlass:
        extra_compile_args["sycl"] = [
            '-DCUTLASS_ENABLE_SYCL=on',
            '-DDPCPP_SYCL_TARGET=intel_gpu_pvc', 
        ]

    use_cuda = torch.cuda.is_available() and (
        CUDA_HOME is not None or ROCM_HOME is not None
    )
    extension = CUDAExtension if use_cuda else CppExtension
    if SYCL_HOME:
        extension = SyclExtension

    extensions_dir = f"{package_name}/csrc"
    sources = list(glob.glob(os.path.join(extensions_dir, "**/*.cpp"), recursive=True))
    sources += list(glob.glob(os.path.join(extensions_dir, "**/*.sycl"), recursive=True))

    # Option 1: since torch xpu ops didn't expose these helpers to submit, we hardcode the path here
    # may copy it into the projection
    include_dirs=[
        "/4T-720/leslie/inductor/pytorch/third_party/torch-xpu-ops/src/",
    ]
    if use_cutlass:
        include_dirs += [
            os.path.join(Path(f"{package_name}").resolve(), "include/cutlass/include/"),
        ]
        
    
    # Option 2: Copy the comm header into this project and include it
    # include_dirs=[f"{Path(__file__).parent.resolve()}/{package_name}/csrc/sycl",]
    
    ext = extension(
        f"{package_name}._C",  # module name of extension
        sources,  # C++ source code files
        py_limited_api=True,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,  # extra compile flag
        extra_link_args=extra_link_args,
    )
    ext_modules = [ext,]

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
        build_dirs = ['build', 'dist', f'{package_name}.egg-info']
        for build_dir in build_dirs:
            if os.path.exists(build_dir):
                shutil.rmtree(build_dir)
        
        # Clean all the so of extension
        so_dir = Path(__file__).parent/package_name
        so_files = list(so_dir.glob("*.so"))
        for so_file in so_files:
            if os.path.exists(so_file.resolve()):
                os.remove(so_file.resolve())

setup(
    name=package_name,  # Project name
    version="0.1.0",  # Initial version
    author="Leslie Fang",
    packages=find_packages(),  # Automatically find all packages
    ext_modules=get_extensions(),
    cmdclass={
        'clean': CustomCleanCommand,  # Register our custom clean command
        'build_ext': BuildExtension,
    },
)