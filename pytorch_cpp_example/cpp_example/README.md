## How to
1. Build Pytorch from source
```
eda2ddb5b06dce13bafd2a745e4634802e4640ef
```

2. Build this test
```
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`
make
```

## Run with 2 threads
```
export OMP_NUM_THREADS=2
numactl -C 0-1 -m 0 ./example-app
```
Got different results at `BS15` and `class1` in single thread and multi-thread count

