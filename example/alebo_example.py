#!/usr/bin/python
"""
==================================================
LASSOBench
High-Dimensional Hyperparameter
Optimization Benchmark
Contact: kenan.sehic@cs.lth.se

Example: ALEBO on synthetic benchmark
=================================================
"""
import numpy as np
import LASSOBench

from ax.modelbridge.strategies.alebo import ALEBOStrategy
from ax.service.managed_loop import optimize

from joblib import Parallel, delayed
import multiprocessing

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


# define synt bench
synt_bench = LASSOBench.Synt_bench(pick_bench='synt_high_eff_bench')
d = synt_bench.n_features


# define objective function for parallel
def run_alebo(low_d, n_doe, n_total):
    """
    Running ALEBO in parallel for a synt bench
    """

    def loss_fun(config):
        x = np.empty((d,))
        for xi in range(d):
            x[xi] = config["x" + str(xi)]

        objective_value = synt_bench.evaluate(input_config=x)
        return {"objective": (objective_value, 0.0)}

    parameters = [
        {"name": "x0", "type": "range",
         "bounds": [-1.0, 1.0], "value_type": "float"},
    ]
    parameters.extend([
        {"name": f"x{i}", "type": "range",
         "bounds": [-1.0, 1.0], "value_type": "float"}
        for i in range(1, d)
    ])

    def run_alebo_parallel(n_seed):
        """Alebo objective for parallel run
            Arg: n_seed - seed number"""

        alebo_strategy = ALEBOStrategy(D=d, d=low_d,
                                       init_size=n_doe)

        best_config, _, experiment, _ = optimize(
            parameters=parameters,
            experiment_name="alebo_test",
            objective_name="objective",
            evaluation_function=loss_fun,
            minimize=True,
            total_trials=n_total,
            random_seed=n_seed,
            generation_strategy=alebo_strategy,
        )

        objectives = np.array([trial.objective_mean for trial in experiment.trials.values()])

        start_time = list(experiment.trials.values())[0]
        start_time = start_time.time_created

        time_alebo = np.array([(trial.time_completed - start_time).total_seconds() for trial in experiment.trials.values()])

        config = np.array([trial.arm.parameters.values() for trial in experiment.trials.values()])

        config_all = np.empty((n_total, d))

        for i in range(n_total):
            config_all[i, :] = np.array(list(config[i]))

        return [objectives, time_alebo, config_all]

    return run_alebo_parallel


# run ALEBO
de = 10
n_doe = 20
n_total = 60
n_repeat = 5

fun_alebo = run_alebo(low_d=de, n_doe=n_doe, n_total=n_total)
random_seeds = np.random.randint(200, size=n_repeat)
num_cores = multiprocessing.cpu_count()
par_res = Parallel(n_jobs=num_cores)(
    delayed(fun_alebo)(n_seed) for n_seed in random_seeds)

loss_alebo = np.empty((n_total, n_repeat))
time_alebo = np.empty((n_total, n_repeat))
mspe_alebo = np.empty((n_total, n_repeat))
fscore_alebo = np.empty((n_total, n_repeat))

for i in range(n_repeat):
    loss_alebo[:, i] = par_res[i][0]
    time_alebo[:, i] = par_res[i][1]
    for j in range(n_total):
        config = par_res[i][2]
        mspe_alebo[j, i], fscore_alebo[j, i] = synt_bench.test(input_config=config[j, :])

# plot
marker = ['p', 'X', 'o']
c_list = sns.color_palette("colorblind")

plt.close('all')
fig = plt.figure(figsize=(26, 10.12), constrained_layout=True)
spec2 = gridspec.GridSpec(nrows=3, ncols=1, figure=fig)

f_ax1 = fig.add_subplot(spec2[0, 0])
f_ax1.plot(range(1, n_total + 1),
           np.mean(loss_alebo, axis=1), '--', color=c_list[0], linewidth=3,
           marker=marker[0], markersize=10, label=r'Average ALEBO with $d_e=10$')
plt.legend(loc='best', fontsize=18)
plt.title('Loss', fontsize=18)
plt.xlabel('Iterations', fontsize=18)
plt.xlim(1, n_total + 1)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.grid(True)

f_ax2 = fig.add_subplot(spec2[1, 0])
f_ax2.plot(range(1, n_total + 1),
           np.mean(mspe_alebo, axis=1), '--', color=c_list[1], linewidth=3,
           marker=marker[1], markersize=10, label=r'Average ALEBO with $d_e=10$')
plt.legend(loc='best', fontsize=18)
plt.title('MSPE divided with oracle', fontsize=18)
plt.xlabel('Iterations', fontsize=18)
plt.xlim(1, n_total + 1)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.grid(True)

f_ax3 = fig.add_subplot(spec2[2, 0])
f_ax3.plot(range(1, n_total + 1),
           np.mean(fscore_alebo, axis=1), '--', color=c_list[2], linewidth=3,
           marker=marker[2], markersize=10, label=r'Average ALEBO with $d_e=10$')
plt.legend(loc='best', fontsize=18)
plt.title('Fscore', fontsize=18)
plt.xlabel('Iterations', fontsize=18)
plt.xlim(1, n_total + 1)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.grid(True)

plt.show()

