import torch

def test_binary_folding():
    class M(torch.nn.Module):
        def __init__(
            self,
        ):
            super().__init__()
            self.conv = torch.nn.Conv2d(128, 128, 3)
            self.bn1 = torch.nn.BatchNorm2d(128)
            self.bn2 = torch.nn.BatchNorm2d(128)

        def forward(self, x):
            x = self.conv(self.bn1(x))
            return self.bn2(x)
    mod = M().eval()
    v = torch.randn((1, 128, 8, 8), dtype=torch.float32, requires_grad=False).add(1)

    with torch.no_grad():
        ref_res = mod(v)
        cfn = torch.compile(mod)
        res = cfn(v)
        print(torch.allclose(ref_res, res), flush=True)


if __name__ == '__main__':
    # test_binary_folding()
    res_pred_masks = torch.load("res_pred_masks.pt")
    ref_pred_masks = torch.load("ref_pred_masks.pt")
    # print(res_pred_masks, flush=True)
    # print(ref_pred_masks, flush=True)
    # print(res_pred_masks.size(), flush=True)

    for boxes in range(res_pred_masks.size(0)):
        # print(boxes, flush=True)
        # print(res_pred_masks[boxes, :, :], flush=True)
        # print(ref_pred_masks[boxes, :, :], flush=True)

        for h in range(res_pred_masks.size(1)):
            for w in range(res_pred_masks.size(2)):
                if res_pred_masks[boxes, h, w] != ref_pred_masks[boxes, h, w]:
                    print("Error", flush=True)
                    print(boxes, h, w, res_pred_masks[boxes, h, w], ref_pred_masks[boxes, h, w], flush=True)
                    break