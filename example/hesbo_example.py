#!/usr/bin/python
"""
==================================================
LASSOBench
High-Dimensional Hyperparameter
Optimization Benchmark
Contact: kenan.sehic@cs.lth.se

Example: HesBO on synthetic benchmark
=================================================
"""
import numpy as np
import lasso_bench
from hesbo_lib import run_main as hesbo_run

from joblib import Parallel, delayed
import timeit

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# define synt bench
synt_bench = lasso_bench.SyntheticBenchmark(pick_bench='synt_low_eff_bench')
d = synt_bench.n_features


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
            obj_value = synt_bench.evaluate(input_config=config)
            obj_value = np.array([obj_value])
            obj_value = obj_value.reshape(1, 1)
            config_all = config
            time_stop = timeit.default_timer()
        else:
            obj_value[j] = synt_bench.evaluate(input_config=config)
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
        mspe_hesbo = np.empty((n_total, n_repeat))
        fscore = np.empty((n_total, n_repeat))
        config_all = np.empty((n_total, d, n_repeat))

        for i in range(n_repeat):
            loss[:, i] = np.squeeze(par_res[i][1])
            elapsed[:, i] = np.squeeze(par_res[i][0])
            config_par = par_res[i][2]
            for j in range(n_total):
                mspe_hesbo[j, i], fscore[j, i] = synt_bench.test(input_config=config_par[j, :])
    else:
        if n_repeat > 1:
            random_seeds = np.random.randint(200000000, size=n_repeat)
            loss = np.empty((n_total, n_repeat))
            elapsed = np.empty((n_total, n_repeat))
            mspe_hesbo = np.empty((n_total, n_repeat))
            fscore = np.empty((n_total, n_repeat))
            config_all = np.empty((n_total, d, n_repeat))

            for i in range(n_repeat):
                _, elapsed0, _, loss0, _, config0 = hesbo_run(low_dim=eff_dim, high_dim=d, initial_n=n_doe,
                                                              total_itr=n_total - n_doe, test_func=evaluate_hesbo, ARD=ARD,
                                                              n_seed=random_seeds[i])
                loss[:, i] = loss0[:, 0]
                elapsed[:, i] = elapsed0[0, :]
                for j in range(n_total):
                    mspe_hesbo[j, i], fscore[j, i] = synt_bench.test(input_config=config0[j, :])
        else:
            _, elapsed, _, loss, _, config = hesbo_run(
                low_dim=eff_dim, high_dim=d, initial_n=n_doe,
                total_itr=n_total - n_doe, test_func=evaluate_hesbo,
                ARD=ARD, n_seed=n_seed)
            mspe_hesbo = np.empty((n_total,))
            fscore = np.empty((n_total,))
            config_all = np.empty((n_total, d))
            for i in range(n_total):
                mspe_hesbo[j, i], fscore[j, i] = synt_bench.test(input_config=config[i, :])

    return -loss, mspe_hesbo, fscore, elapsed

if __name__ == '__main__':
    # run Hesbo parallel on synt
    initial_desing = 4
    total_steps = 12
    eff_dim = 2
    n_repeat = 5
    n_jobs = 5

    loss_hesbo, mspe_hesbo, fscore_hesbo, time_hesbo = run_hesbo(
        eff_dim=eff_dim, n_doe=initial_desing, n_total=total_steps, ARD=True,
        n_repeat=n_repeat, n_seed=42, n_jobs=n_jobs)

    # plot
    marker = ['p', 'X', 'o']
    c_list = sns.color_palette("colorblind")

    plt.close('all')
    fig = plt.figure(figsize=(26, 10.12), constrained_layout=True)
    spec2 = gridspec.GridSpec(nrows=3, ncols=1, figure=fig)

    f_ax1 = fig.add_subplot(spec2[0, 0])
    f_ax1.plot(range(1, total_steps + initial_desing + 1),
               np.mean(loss_hesbo, axis=1), '--', color=c_list[0], linewidth=3,
               marker=marker[0], markersize=10, label=r'Average Hesbo with $d_e=2$')
    plt.legend(loc='best', fontsize=18)
    plt.title('Loss', fontsize=18)
    plt.xlabel('Iterations', fontsize=18)
    plt.xlim(1, total_steps + initial_desing + 1)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.grid(True)

    f_ax2 = fig.add_subplot(spec2[1, 0])
    f_ax2.plot(range(1, total_steps + initial_desing + 1),
               np.mean(mspe_hesbo, axis=1), '--', color=c_list[1], linewidth=3,
               marker=marker[1], markersize=10, label=r'Average Hesbo with $d_e=2$')
    plt.legend(loc='best', fontsize=18)
    plt.title('MSPE divided with oracle', fontsize=18)
    plt.xlabel('Iterations', fontsize=18)
    plt.xlim(1, total_steps + initial_desing + 1)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.grid(True)

    f_ax3 = fig.add_subplot(spec2[2, 0])
    f_ax3.plot(range(1, total_steps + initial_desing + 1),
               np.mean(fscore_hesbo, axis=1), '--', color=c_list[2], linewidth=3,
               marker=marker[2], markersize=10, label=r'Average Hesbo with $d_e=2$')
    plt.legend(loc='best', fontsize=18)
    plt.title('Fscore', fontsize=18)
    plt.xlabel('Iterations', fontsize=18)
    plt.xlim(1, total_steps + initial_desing + 1)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.grid(True)

    plt.show()

    # END
