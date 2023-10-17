## Introduction
Due to the BN numerical differences
* https://github.com/pytorch/pytorch/issues/111192
* https://github.com/pytorch/pytorch/issues/111384

We have 2 solutions to run QAT

### Solution 1
* Step 1: QAT training on CUDA
    * **Note** Do QAT on CUDA due to this issue: https://github.com/pytorch/pytorch/issues/111192
* Step 2: Convert and Lowering on CPU

#### Step 1: Do QAT training on CUDA and save the checkpoint
```
python x86inductorquantizer_qat_acc_cuda.py
```

Detail steps:

* 1.1 Load model and to CUDA
* 1.2 Prepare Model with X86InductorQuantizer
* 1.3 QAT Training Loop
* 1.4 Save the checkpoint

#### Step 2: Convert and Lowering into X86 CPU Inductor
* Accuracy
```
TORCHINDUCTOR_FREEZING=1 python x86inductorquantizer_qat_acc_with_checkpoint.py
```
* Performance
```
TORCHINDUCTOR_FREEZING=1 python test_jira_MLDL_836.py
```

Detail steps:

* Load model
* Prepare Model with X86InductorQuantizer
* Load the QAT checkpoint with weights
* Convert PT2E
* Torch.compile lowers to Inductor
* Validation Loop

### Solution 2
Directly QAT on CPU
* Performance:
```
TORCHINDUCTOR_FREEZING=1 python test_jira_MLDL_836.py
```

* Accuracy:
```
TORCHINDUCTOR_FREEZING=1 python x86inductorquantizer_qat_acc.py 
```
