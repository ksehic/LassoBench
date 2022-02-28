#!/usr/bin/python
"""
==================================================
LassoBench
High-Dimensional Hyperparameter
Optimization Benchmark
Contact: kenan.sehic@cs.lth.se

Example: HesBO on synthetic and real benchmark
=================================================
"""
import numpy as np
import LassoBench
from hesbo_lib import RunMain as hesbo_run

from joblib import Parallel, delayed
import timeit

import multiprocessing

# select benchmark
pick_bench = 'rcv1'

if pick_bench == 'synt_hard':
    # define synt bench with 1000 features
    synt_bench = LassoBench.SyntheticBenchmark(pick_bench='synt_high')
    d = synt_bench.n_features
elif pick_bench == 'rcv1':
    # define real bench
    real_bench = LassoBench.RealBenchmark(pick_data='rcv1')
    d = real_bench.n_features


# prepare objective function
def evaluate_hesbo(config_x):
    j = 0
    if len(config_x.shape) != 1:
        obj_value = np.empty((config_x.shape[0], ))
        config_all = np.empty((config_x.shape[0], d))
        time_stop = np.empty((config_x.shape[0], ))
    for i in config_x:
        config = np.empty((d,))
        for z in range(d):
            config[z] = i[z]

        if len(config_x.shape) == 1:
            # obj_value = synt_bench.evaluate(input_config=config)
            obj_value = real_bench.evaluate(input_config=config)
            obj_value = np.array([obj_value])
            obj_value = obj_value.reshape(1, 1)
            config_all = config
            time_stop = timeit.default_timer()
        else:
            # obj_value[j] = synt_bench.evaluate(input_config=config)
            obj_value[j] = real_bench.evaluate(input_config=config)
            time_stop[j] = timeit.default_timer()
            config_all[j, :] = config
            j = j + 1

    if len(config_x.shape) != 1:
        obj_value = obj_value.reshape(config_x.shape[0], 1)

    return -obj_value, config_all, time_stop


# define HESBO function
def run_hesbo(eff_dim, n_doe, n_total, ARD=True, n_repeat=0, n_seed=42, n_jobs=1):

    n_total = n_total + n_doe

    if n_jobs > 1:

        def run_parallel_hesbo(low_dim, high_dim, initial_n, total_itr, test_func, ARD):
            def parallel_hesbo(n_seed):
                _, elapsed0, _, loss0, _, config0 = hesbo_run(low_dim=low_dim, high_dim=high_dim, initial_n=initial_n,
                                                              total_itr=total_itr, test_func=test_func, ARD=ARD,
                                                              n_seed=n_seed)
                return elapsed0, loss0, config0
            return parallel_hesbo

        hesbo_objective = run_parallel_hesbo(low_dim=eff_dim, high_dim=d, initial_n=n_doe,
                                                total_itr=n_total - n_doe, test_func=evaluate_hesbo, ARD=ARD)
        random_seeds = np.random.randint(200000000, size=n_repeat)
        par_res = Parallel(n_jobs=n_jobs)(
            delayed(hesbo_objective)(n_seed) for n_seed in random_seeds)

        loss = np.empty((n_total, n_repeat))
        elapsed = np.empty((n_total, n_repeat))
        for i in range(n_repeat):
            loss[:, i] = np.squeeze(par_res[i][1])
            elapsed[:, i] = np.squeeze(par_res[i][0])
    else:
        if n_repeat > 1:
            random_seeds = np.random.randint(200000000, size=n_repeat)
            loss = np.empty((n_total, n_repeat))
            elapsed = np.empty((n_total, n_repeat))

            for i in range(n_repeat):
                _, elapsed0, _, loss0, _, _ = hesbo_run(low_dim=eff_dim, high_dim=d, initial_n=n_doe,
                                                              total_itr=n_total - n_doe, test_func=evaluate_hesbo, ARD=ARD,
                                                              n_seed=random_seeds[i])
                loss[:, i] = loss0[:, 0]
                elapsed[:, i] = elapsed0[0, :]
        else:
            _, elapsed, _, loss, _, _ = hesbo_run(
                low_dim=eff_dim, high_dim=d, initial_n=n_doe,
                total_itr=n_total - n_doe, test_func=evaluate_hesbo,
                ARD=ARD, n_seed=n_seed)

    return -loss, elapsed


if __name__ == '__main__':

    n_jobs = multiprocessing.cpu_count()
    total_steps = 1000
    d_low = np.array([2, 5, 10, 20])
    n_repeat = 30

    loss_hesbo_n = []
    time_hesbo_n = []

    for i in range(4):

        de = d_low[i]

        # run Hesbo parallel on synt
        initial_desing = de + 1

        loss_hesbo, time_hesbo = run_hesbo(
            eff_dim=de, n_doe=initial_desing, n_total=total_steps, ARD=True,
            n_repeat=n_repeat, n_seed=42, n_jobs=n_jobs)

        loss_hesbo_n.append(loss_hesbo)
        time_hesbo_n.append(time_hesbo)

    # END
