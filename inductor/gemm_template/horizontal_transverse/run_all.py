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
    # typical case
    (2048, 2048, 8192),
]

shapes = [
    (8064, 4096, 11008),
    (8064, 32000, 4096),
]
test_vectical = True
test_horizontal = True

def test(cmd, res, log_file):
    os.system("rm -rf {}".format(log_file))
    # if test_only:
    #     cmd += " --warmup 5 --run 10 --count 1"
    os.system("{0} 2>&1 | tee {1}".format(cmd, log_file))
    with open(log_file, 'r', newline='') as file:
        lines = file.readlines()
        for line in lines:
            if "GEMM" in line:
                time = line.strip().split("compile:")[1].strip().split("ms")[0].strip()
                res.append(time)
                break

if __name__ == "__main__":
    log_file = "./test_single.log"
    prefix = "rm -rf /tmp/torchinductor_leslie/* && rm -rf torch_compile_debug/* && numactl -C 96-127 -m 3 python -u bench_linear.py "
    with open("result.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        field = ["M", "N", "K", "vertical(ms)", "horizontal(ms)"]
        writer.writerow(field)
        for shape in shapes:
            result = []
            m, n, k = shape
            result.extend((m, n, k))

            if test_vectical:
                cmd = prefix + " --verbose --batch-size {0} --in-features {1} --out-features {2} --cpp ".format(m, k, n)
                test(cmd, result, log_file)
            else:
                result.append(0.0)

            if test_horizontal:
                cmd = prefix + " --verbose --batch-size {0} --in-features {1} --out-features {2} --cpp --horizontal".format(m, k, n)
                test(cmd, result, log_file)
            else:
                result.append(0.0)
            writer.writerow(result)