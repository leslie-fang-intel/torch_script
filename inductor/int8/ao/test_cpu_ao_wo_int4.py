import torch
import torchtune

def test_int4_wo_on_torch_tune_model_cpu():
        from torchtune.models.llama2 import llama2_7b
        model = llama2_7b()
        model.to(device="cpu")
        from torchao.quantization.quant_api import change_linear_weights_to_int4_woqtensors
        vocab_size = 32000
        bsz = 2
        seq_len = 100
        example_inputs = torch.randint(0, vocab_size, (bsz, seq_len)).to(device="cpu")
        import time
        start = time.time()
        model(example_inputs)
        end = time.time()
        print("fp32 time:", end - start)

        change_linear_weights_to_int4_woqtensors(model)

        start = time.time()
        model(example_inputs)
        end = time.time()
        print("unlowered cpu time:", end - start)
        with torch.no_grad():
            model = torch.compile(model, mode="max-autotune")
            for _ in range(2):
                # warm up
                model(example_inputs)

            ITER = 10
            t = 0.0
            for _ in range(ITER):
                start = time.time()
                model(example_inputs.to(device="cpu"))
                end = time.time()
                print("lowered cpu time:", end - start)
                t += end - start
            print("avg cpu time:", t / ITER)

if __name__ == "__main__":
    test_int4_wo_on_torch_tune_model_cpu()
