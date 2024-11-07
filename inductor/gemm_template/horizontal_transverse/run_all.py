import torch
import csv
import os

# M, N, K
shapes = [
    (4, 12288, 4096),
    (4, 4096, 11008),
    (4, 32000, 4096),
    (16, 12288, 4096),
    (16, 4096, 11008),
    (16, 32000, 4096),
    (1024, 12288, 4096),
    (1024, 4096, 11008),
    (1024, 32000, 4096),
    (4096, 12288, 4096),
    (4096, 4096, 11008),
    (4096, 32000, 4096),
    (2016, 12288, 4096),
    (2016, 4096, 11008),
    (2016, 32000, 4096),
    (8064, 12288, 4096),
    (8064, 4096, 11008),
    (8064, 32000, 4096),
]

def parse_log(log_file, result):
    with open(log_file, 'r', newline='') as file:
        lines = file.readlines()
        for line in lines:
            if "GEMM" in line:
                time = line.strip().split("compile:")[1].strip().split("ms")[0].strip()
                result.append(time)
                break

if __name__ == "__main__":
    log_file = "./test_single.log"
    prefix = "rm -rf /tmp/torchinductor_leslie/* && rm -rf torch_compile_debug/* && numactl -C 96-127 -m 3 python -u bench_linear.py "
    with open("result.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        field = ["M", "N", "K", "perf"]
        writer.writerow(field)
        for shape in shapes:
            result = []
            os.system("rm -rf {}".format(log_file))
            m, n, k = shape
            result.extend((m, n, k))
            cmd = prefix + "--verbose --batch-size {0} --in-features {1} --out-features {2} --horizontal".format(m, k, n)
            cmd += " 2>&1 | tee {}".format(log_file)
            print("cmd is: {}".format(cmd), flush=True)
            os.system(cmd)
            parse_log(log_file, result)
            writer.writerow(result)