#!/usr/bin/python
"""
==================================================
LASSOBench
High-Dimensional Hyperparameter
Optimization Benchmark
Contact: kenan.sehic@cs.lth.se

Tutorial: Random search on different benchmarks
=================================================
"""
import numpy as np
from tqdm import tqdm
import LASSOBench

# prepare benchmark
synt_bench = LASSOBench.Synt_bench(pick_bench='synt_high_noise_bench')
d = synt_bench.n_features

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
    mspe[i], fscore[i], _ = synt_bench.test(random_config[i, :])

# run lassocv
loss_lcv, mspe_lcv, fscore_lcv, config_lcv, time_lcv = synt_bench.run_LASSOCV()

# run sparse-ho
loss_sho, mspe_sho, fscore_sho, config_sho, reg_sho, time_sho = synt_bench.run_sparseho(n_steps=30)

# define real world and run on cheap fidelity
real_bench = LASSOBench.Realworld_bench(pick_data='rcv1', mf_opt='multi_source_bench')
d = real_bench.n_features

# random config
random_config = np.random.uniform(low=-1.0, high=1.0, size=(N, d))

# run random search on cheapest model
fidelity_pick = 0
loss_cheap = np.empty((N,))

for i in tqdm(range(N), ascii=True,
              desc='New config'):
    loss_cheap[i] = real_bench.fidelity_evaluate(
        random_config[i, :], index_fidelity=fidelity_pick)

# END
