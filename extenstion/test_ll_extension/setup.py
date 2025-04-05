from setuptools import setup, find_packages, Extension
import glob
import os
import pybind11

extra_link_args = []
extensions_dir = "test_ll_extension/csrc/"
sources = list(glob.glob(os.path.join(extensions_dir, "**/*.cpp"), recursive=True))

ext_modules = [
    Extension(
        "test_ll_extension.example",  # 生成的 Python 模块名
        sources,  # C++ 源文件路径
        include_dirs=[pybind11.get_include()], 
        extra_compile_args=["-std=c++11"],  # 额外的编译参数
        extra_link_args=extra_link_args,
    )
]

setup(
    name="test_ll_extension",  # Project name
    version="0.1.0",  # Initial version
    author="Leslie Fang",
    packages=find_packages(),  # Automatically find all packages
    ext_modules=ext_modules,
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