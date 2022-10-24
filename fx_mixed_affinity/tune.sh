export LD_PRELOAD="/pytorch/leslie/jemalloc/lib/libjemalloc.so":$LD_PRELOAD
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export LD_PRELOAD=$LD_PRELOAD:/pytorch/leslie/anaconda3/pkgs/intel-openmp-2021.3.0-h06a4308_3350/lib/libiomp5.so

export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0
# sudo pip install opentuner==0.8.8
#numactl --physcpubind=0-55 --membind=0 python test_mixed_affinity_opentuner.py
numactl --physcpubind=0-55 --membind=0 python test_mixed_affinity_tuner.py
