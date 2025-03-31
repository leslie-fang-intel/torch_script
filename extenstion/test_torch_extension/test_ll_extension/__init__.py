import torch
import logging

try:
    from pathlib import Path

    so_files = list(Path(__file__).parent.glob("_C*.so"))
    if len(so_files) > 0:
        assert len(so_files) == 1, f"Expected one _C*.so file, found {len(so_files)}"
        torch.ops.load_library(str(so_files[0]))
        from . import ops

    # # The following library contains CPU kernels from torchao/experimental
    # # They are built automatically by ao/setup.py if on an ARM machine.
    # # They can also be built outside of the torchao install process by
    # # running the script `torchao/experimental/build_torchao_ops.sh <aten|executorch>`
    # # For more information, see https://github.com/pytorch/ao/blob/main/torchao/experimental/docs/readme.md
    # experimental_lib = list(Path(__file__).parent.glob("libtorchao_ops_aten.*"))
    # if len(experimental_lib) > 0:
    #     assert (
    #         len(experimental_lib) == 1
    #     ), f"Expected at most one libtorchao_ops_aten.* file, found {len(experimental_lib)}"
    #     torch.ops.load_library(str(experimental_lib[0]))
except:
    logging.debug("Skipping import of cpp extensions")