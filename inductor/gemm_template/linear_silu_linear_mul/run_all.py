import os
import csv

shapes = [
    # BS: 1, input: 1024, output: 128 First token
    (4 * 1024, 11008, 4096),
    # BS: 1, input: 1024, output: 128 Next token
    (4, 11008, 4096),
    # (4, 4096, 4096)
    # (4, 32000, 4096)

    # BS: 4, input: 1024, output: 128 First token
    (4 * 4 * 1024, 11008, 4096),
    # BS: 4, input: 1024, output: 128 Next token
    (4 * 4, 11008, 4096),

    # BS: 1, input: 2016, output: 32 First token
    (4 * 2016, 11008, 4096),
    # BS: 4, input: 2016, output: 32 First token
    (4 * 4 * 2016, 11008, 4096),
]

# shapes = [
#     # BS: 1, input: 1024, output: 128 Next token
#     (4, 11008, 4096),

#     # BS: 4, input: 2016, output: 32 First token
#     # (4 * 4 * 2016, 11008, 4096),
# ]

test_ipex = True
test_aten = True
test_template = True
test_tempate_fusion = True

log_file = "/home/leslie/lz/torch_script/inductor/gemm_template/linear_silu_linear_mul/test.log"
base_cmd_wo_shape = "rm -rf /tmp/torchinductor_leslie/* && rm -rf torch_compile_debug/* && clear && numactl -C 96-127 -m 3 python -u bench_linear.py --verbose --batch-size {0} --in-features {1} --out-features {2}"

test_only = False  # use small steps

def test(cmd, res):
    os.system("rm -rf {}".format(log_file))
    if test_only:
        cmd += " --warmup 5 --run 10 --count 1"
    os.system("{0} 2>&1 | tee {1}".format(cmd, log_file))
    with open(log_file, 'r', newline='') as file:
        lines = file.readlines()
        for line in lines:
            if "GEMM" in line:
                time = line.strip().split("compile:")[1].strip().split("ms")[0].strip()
                res.append(time)
                break

if __name__ == "__main__":
    with open('res.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        field = ["M", "N", "K", "bias_gate", "bias_up", "IPEX", "Torch OneDNN", "Torch GEMM Template wo silu_mul fusion", "Torch GEMM Template w silu_mul fusion"]
        writer.writerow(field)
        for shape in shapes:
            M, N, K = shape
            res = [M, N, K, False, False]
            base_cmd = base_cmd_wo_shape
            base_cmd = base_cmd.format(M, N, K)

            # Perf of IPEX
            if test_ipex:
                cmd = base_cmd + " --ipex"
                test(cmd, res)
            else:
                res.append(0.0)
            
            # Perf of Torch ATEN
            if test_aten:
                cmd = base_cmd + " --aten"
                test(cmd, res)
            else:
                res.append(0.0)

            # Perf of Torch Template
            if test_template:
                cmd = base_cmd
                test(cmd, res)
            else:
                res.append(0.0)

            # Perf of Torch Template with Fusion
            if test_tempate_fusion:
                cmd = base_cmd + " --fusion"
                test(cmd, res)        
            else:
                res.append(0.0)

            writer.writerow(res)