rm -rf /tmp/torchinductor_leslie/* && rm -rf torch_compile_debug/* && clear && numactl -C 96-127 -m 3 python -u bench_linear.py --verbose --batch-size 2048 --in-features 8192 --out-features 2048 --cpp
rm -rf /tmp/torchinductor_leslie/* && rm -rf torch_compile_debug/* && clear && numactl -C 96-127 -m 3 python -u bench_linear.py --verbose --batch-size 2048 --in-features 8192 --out-features 2048 --cpp --horizontal