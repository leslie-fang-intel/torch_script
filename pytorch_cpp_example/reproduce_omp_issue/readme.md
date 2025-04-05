## ISSUE 
```
Install pytorch 2.6

pip install torch --index-url https://download.pytorch.org/whl/cpu

unset LD_PRELOAD ## remove iomp link
```

## Build testcase
```
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`
make
```

## PyTorch 2.6 release
### Run without GOMP_CPU_AFFINITY
```
./example-app      
omp_get_max_threads() is: 224
```

### Run with GOMP_CPU_AFFINITY
```
GOMP_CPU_AFFINITY=0-95 ./example-app

omp_get_max_threads() is: 1
```

## PyTorch build from src and no iomp
```
commit: d4496346b901e9a4c3993bf6b2054014c7c0b731

unset LD_PRELOAD && python setup.py develop

1. 修改 pytorch code, 优先link gomp 而不是 omp
https://github.com/pytorch/pytorch/pull/138834/files

2. 修改 release package的软链接
* rm -rf /localdisk/leslie/miniforge/envs/pytorch_lz/lib/python3.10/site-packages/torch/lib/libgomp-a34b3233.so.1
* ln -sf /localdisk/leslie/miniforge/envs/pytorch_lz/lib/libgomp.so.1 /localdisk/leslie/miniforge/envs/pytorch_lz/lib/python3.10/site-packages/torch/lib/libgomp-a34b3233.so.1

重新编译 test case

```

### Run without GOMP_CPU_AFFINITY
```
./example-app      
omp_get_max_threads() is: 224
```

### Run with GOMP_CPU_AFFINITY
```
GOMP_CPU_AFFINITY=0-95 ./example-app
omp_get_max_threads() is: 224
```