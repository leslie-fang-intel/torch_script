A PyTorch Extension based on CUDA/CUTLASS

## Build
```
set CURRENT_DIR=$(pwd)

* Clone cutlass
cd torch_cuda_extension/csrc/include && git clone https://github.com/NVIDIA/cutlass.git && cd $CURRENT_DIR

* Build the package
python setup.py develop
```

### Verified
``` 
PyTorch e6afb51805d88547803252e4fb1a2757624d590c
CUDA 12.6
```

## Useage
Python based
```
cd tests
python test_extended_add.py

```
