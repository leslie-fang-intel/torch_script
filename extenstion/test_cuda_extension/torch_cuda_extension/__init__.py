import torch
import logging

try:
    from pathlib import Path

    so_files = list(Path(__file__).parent.glob("_C*.so"))
    if len(so_files) > 0:
        assert len(so_files) == 1, f"Expected one _C*.so file, found {len(so_files)}"
        torch.ops.load_library(str(so_files[0]))
        from . import ops

except Exception as e:
    logging.error("Failed to import of cpp extensions of torch_cuda_extension due to: {}".format(e))