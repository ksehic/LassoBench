#!/usr/bin/python
"""
==================================================
LassoBench
High-Dimensional Hyperparameter
Optimization Benchmark
Contact: kenan.sehic@cs.lth.se

Tutorial: Random search on different benchmarks
          Run LassoCV, Sparse-HO and Multi-start Sparse-HO
          Test multi-fidelity approach
=================================================
"""
import numpy as np
from tqdm import tqdm
import LassoBench


# prepare noiseless benchmark
synt_bench = LassoBench.SyntheticBenchmark(pick_bench='synt_simple')
d = synt_bench.n_features

# prepare noisy benchmark
synt_bench_noisy = LassoBench.SyntheticBenchmark(pick_bench='synt_simple', noise=True)
d = synt_bench_noisy.n_features

# generate random config within [-1, 1]
N = 1000

# numpy array
random_config = np.random.uniform(low=-1.0, high=1.0, size=(N, d))

# The objective function for a HPO Algorithm
loss = np.empty((N,))
loss_noisy = np.empty((N,))

for i in tqdm(range(N), ascii=True,
            desc='New config'):
    loss[i] = synt_bench.evaluate(random_config[i, :])
    loss_noisy[i] = synt_bench_noisy.evaluate(random_config[i, :])

# run lassocv
loss_lcv, _, _, time_lcv = synt_bench.run_LASSOCV()
loss_lcv_noisy, _, _, time_lcv_noisy = synt_bench_noisy.run_LASSOCV()

# run sparse-ho
loss_sho, _, _, time_sho = synt_bench.run_sparseho(n_steps=30)
loss_sho_noisy, _, _, time_sho_noisy = synt_bench_noisy.run_sparseho(n_steps=30)

# run multi-start sparse-ho for random configurations
n_starts = 10
loss_msho = []
time_msho = []
loss_msho_noisy = []
time_msho_noisy = []

for i in range(n_starts):
        if i == 0 :
                loss_sho0, _, _, time_sho0 = synt_bench.run_sparseho(n_steps=20)
                loss_sho_noisy0, _, _, time_sho_noisy0 = synt_bench_noisy.run_sparseho(n_steps=20)
        else:
                lambda_random = np.random.uniform(low=-1.0, high=1.0)
                loss_sho0, _, _, time_sho0 = synt_bench.run_sparseho(
                        n_steps=20, init_point=np.ones((d,))*lambda_random)
                loss_sho_noisy0, _, _, time_sho_noisy0 = synt_bench_noisy.run_sparseho(
                        n_steps=20, init_point=np.ones((d,))*lambda_random)
        loss_msho.append(loss_sho0)
        loss_msho_noisy.append(loss_sho_noisy0)

# For each config, we can measure MSE on the test data
# and additionally the F-score but only for a synth bench via .test
random_config = np.random.uniform(low=-1.0, high=1.0, size=(d,))
mse_test, fscore = synt_bench.test(random_config)

# define real world bench and run on the cheapest fidelity l=0
real_bench = LassoBench.RealBenchmark(pick_data='rcv1', mf_opt='discrete_fidelity')
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

# MSE on the test data
mse_test = real_bench.test(random_config[0, :])

# END
