import torch
from torch._inductor import codecache, config, metrics, test_operators

if __name__ == "__main__":
    # supported_vec = codecache.supported_vec_isa_list
    print("valid_vec_isa is: {}".format(codecache.valid_vec_isa_list()), flush=True)

