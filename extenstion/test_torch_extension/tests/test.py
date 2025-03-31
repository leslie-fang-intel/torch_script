import torch
import torchao
import torchao.ops
# import test_ll_extension

if __name__ == "__main__":
    shape = (4096, 11008)
    # t = torch.randint(0, 16, dtype=torch.int, size=shape, device="cuda")
    # t = (t[::, ::2] << 4 | t[::, 1::2]).to(torch.uint8)
    inner_k_tiles = 2
    # packed_w = torch.ops.aten._convert_weight_to_int4pack(t, inner_k_tiles)
    # unpacked = torchao.ops.unpack_tensor_core_tiled_layout(packed_w, inner_k_tiles)

    t = torch.randint(0, 16, dtype=torch.int, size=shape, device="cpu")
    # torch.ops.torchao.toy_op2(t)

    # print(torch._C._dispatch_dump("torchao::toy_op2"), flush=True)
    # print("Registered torchao ops:", dir(torch.ops.torchao))

    import test_ll_extension
    print(test_ll_extension.__file__, flush=True)

    # import test_ll_extension.test_ll_add as test_ll_add
    # res = test_ll_add.test_ll_add(t, t)
    # print("res of test_ll_add is: {}".format(res), flush=True)

    import test_ll_extension.ops

    res2 = torch.ops.torchll.toy_test_ll(t)

    # print("Res is: {}".format(res), flush=True)
    print("Done ----", flush=True)

