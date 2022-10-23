export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0
# sudo pip install opentuner==0.8.8
numactl --physcpubind=0-55 --membind=0 python test_mixed_affinity_opentuner.py
