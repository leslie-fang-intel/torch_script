## Version
* PyTorch Build from src
    * torch 2.5.1 release
* Flashinfer build from src: https://docs.flashinfer.ai/installation.html
    * Commit: fdedc4321e98053e75f970d8b183eddda3aadab0
    * 使用 jit 模式编译
* sglang build from src: `pip install -e "python[all]"`
    * Commit: dce303e279e11258dff57e1c25410b2f570a7a2f

## CMD
```
python3 /home/t/leslie/inductor_quant/sglang/python/sglang/bench_one_batch.py --batch-size 1 --input-len 1024 --output-len 8 --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct --trust-remote-code --device cuda --attention-backend flashinfer --enable-flashinfer-mla
```