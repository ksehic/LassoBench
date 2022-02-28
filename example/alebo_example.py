#!/usr/bin/python
"""
==================================================
LassoBench
High-Dimensional Hyperparameter
Optimization Benchmark
Contact: kenan.sehic@cs.lth.se

Example: ALEBO on synthetic benchmark
=================================================
"""
import numpy as np
import LassoBench

from ax.modelbridge.strategies.alebo import ALEBOStrategy
from ax.service.managed_loop import optimize

from joblib import Parallel, delayed
import multiprocessing

# define objective function for parallel
def run_alebo(low_d, n_doe, n_total, bench):
    """
    Running ALEBO in parallel for a synt bench
    """

    def loss_fun(config):
        x = np.empty((d,))
        for xi in range(d):
            x[xi] = config["x" + str(xi)]

        objective_value = bench.evaluate(input_config=x)
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


if __name__ == '__main__':

    # define synt bench
    synt_bench = LassoBench.SyntheticBenchmark(pick_bench='synt_hard')
    d = synt_bench.n_features

    # ALEBO selected arguments
    de = 50
    n_doe = de + 1
    n_total = 100
    n_repeat = 30

    # run ALEBO
    fun_alebo = run_alebo(low_d=de, n_doe=n_doe, n_total=n_total, bench=synt_bench)
    random_seeds = np.random.randint(200, size=n_repeat)
    num_cores = multiprocessing.cpu_count()
    par_res = Parallel(n_jobs=num_cores)(
        delayed(fun_alebo)(n_seed) for n_seed in random_seeds)

    loss_alebo = np.empty((n_total, n_repeat))
    time_alebo = np.empty((n_total, n_repeat))

    for i in range(n_repeat):
        loss_alebo[:, i] = par_res[i][0]
        time_alebo[:, i] = par_res[i][1]

    # END
