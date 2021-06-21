#!/usr/bin/python
"""
==================================================
LASSOBench
High-Dimensional Hyperparameter
Optimization Benchmark
Contact: kenan.sehic@cs.lth.se

Example: Random search on synthetic benchmark
=================================================
"""
import numpy as np
from tqdm import tqdm
import LASSOBench

# bench dimension
d = 1280

# prepare benchmark
synt_bench = LASSOBench.Synt_bench()

# generate random config within [-1, 1]
N = 1000

# numpy array
random_config = np.random.uniform(low=-1.0, high=1.0, size=(N, d))

# Each config is tested for MSPE and Fscore, while HPO is done on loss
loss = np.empty((N,))
mspe = np.empty((N,))   # MSPE divided by the oracle meaning 1 or less is great
fscore = np.empty((N,))  # between 0 and 1. 1 is the best

for i in tqdm(range(N), ascii=True,
              desc='New config'):
    loss[i] = synt_bench.evaluate(random_config[i, :])
    mspe[i], fscore[i] = synt_bench.test(random_config[i, :])

# run Hesbo
loss_hesbo, mspe_hesbo, fscore_hesbo, time_hesbo = synt_bench.run_hesbo(eff_dim=2, n_doe=20, n_total=40, ARD=True, n_repeat=5, n_seed=42, n_jobs=2)

# real-world example
real_bench = LASSOBench.Realworld_bench(data_pick='rcv1')

# run Hesbo on real
loss_hesbo, mspe_hesbo, time_hesbo = real_bench.run_hesbo(eff_dim=2, n_doe=4, n_total=12, ARD=True, n_repeat=5, n_seed=42, n_jobs=5)

# new experiment with predefined w_true
d=1280
w_true = np.zeros((d,))

# define only two nonelements
w_true[0] = 1
w_true[100] = 10

# prepare benchmark
synt_bench_w = LASSOBench.Synt_bench(n_features=d, w_true=w_true, snr_level=1)

# run lassocv
loss_lcv, mspe_lcv, fscore_lcv, config_lcv, time_lcv = synt_bench_w.run_LASSOCV()

# run hesbo
loss_hesbo, mspe_hesbo, fscore_hesbo, config_hesbo, time_hesbo = synt_bench_w.run_hesbo(eff_dim=2, n_doe=20, n_total=60, ARD=True, n_repeat=5, n_seed=42, n_jobs=5)

# run sparse-ho
loss_sho, mspe_sho, fscore_sho, config_sho, reg_sho, time_sho = synt_bench_w.run_sparseho(n_steps=30)

# random searc only on two elements
# generate random config within [-1, 1]
N = 1000

# numpy array
random_config = np.ones((N, d)) * np.inf
random_config0 = np.random.uniform(low=-1.0, high=1.0, size=(N, 2))
random_config[:, 0] = random_config0[:, 0]
random_config[:, 100] = random_config0[:, 1]

# Each config is tested for MSPE and Fscore, while HPO is done on loss
loss = np.empty((N,))
mspe = np.empty((N,))   # MSPE divided by the oracle meaning 1 or less is great
fscore = np.empty((N,))  # between 0 and 1. 1 is the best
reg_coef = np.empty((N, d))

for i in tqdm(range(N), ascii=True,
              desc='New config'):
    # loss[i] = synt_bench_w.evaluate(random_config[i, :])
    mspe[i], fscore[i], reg_coef[i, :] = synt_bench_w.test(random_config[i, :])

# END
