#!/usr/bin/python
"""
==================================================
LassoBench
High-Dimensional Hyperparameter
Optimization Benchmark
Contact: kenan.sehic@cs.lth.se

Example: TuRBO on synthetic benchmark

You need to install TuRBO following
https://github.com/uber-research/TuRBO

Run in terminal: python turbo_example.py 1
=================================================
"""
from turbo import Turbo1
import numpy as np
import pickle
import LassoBench
import sys


class LassoBenchFunction:
    """
    Select a benchmark from LassoBench.
    """

    def __init__(self, noise=False):
        """
        For synthentic benchmarks, we test TuRBO on the noisy and noiseless case.

        Args:
            noise (boolean): selecting noisy or noiseless benchmark
        """
        self.synt_bench = LassoBench.SyntheticBenchmark(pick_bench='synt_hard', noise=noise)
        dim = self.synt_bench.n_features
        self.dim = dim
        self.lb = -1 * np.ones(dim)
        self.ub = 1 * np.ones(dim)

    def __call__(self, x):

        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)

        obj_value = self.synt_bench.evaluate(x)

        return obj_value


def run_turbo_par(noise):
    """
    Running TuRBO in parallel

    Args:
        noise (boolean): selecting noisy or noiseless benchmark
    """
    def run_turbo(seed):
        """
        Running TuRBO for different seends

        Args:
            seed (int)): selecting different seeds
        """
        np.random.seed(seed)

        f = LassoBenchFunction(noise=noise)

        n_steps = 5000

        turbo1 = Turbo1(
            f=f, # Handle to objective function
            lb=f.lb, # Numpy array specifying lower bounds
            ub=f.ub, # Numpy array specifying upper bounds
            n_init=100, # Number of initial bounds from an Latin hypercube design
            max_evals = n_steps, # Maximum number of evaluations
            batch_size=10, # How large batch size TuRBO uses
            verbose=False, # Print information from each batch
            use_ard=True, # Set to true if you want to use ARD for the GP kernel
            max_cholesky_size=2000, # When we switch from Cholesky to Lanczos
            n_training_steps=50, # Number of steps of ADAM to learn the hypers
            min_cuda=1024, # Run on the CPU for small datasets
            device="cpu", # "cpu" or "cuda"
            dtype="float64", # float64 or float32
            )

        turbo1.optimize()

        X = turbo1.X # Evaluated points
        loss_turbo = turbo1.fX # Observed values

        mspe_turbo = np.empty((n_steps,))
        fscore_turbo = np.empty((n_steps,))
        synt_bench = LassoBench.SyntheticBenchmark(pick_bench='synt_hard', noise=noise)

        for i in range(n_steps):
            mspe_turbo[i], fscore_turbo[i] = synt_bench.test(X[i, :])

        return np.squeeze(loss_turbo), mspe_turbo, fscore_turbo

    return run_turbo


def main_turbo(n_seed):

    n_repeat = 300
    noise_pick = [True, False]
    loss_turbo = []
    mspe_turbo = []
    fscore_turbo = []

    for j in range(2):

        turbo_objective = run_turbo_par(noise=noise_pick[j])
        random_seeds = np.random.randint(200000000, size=n_repeat)
        loss_turbo0, mspe_turbo0, fscore_turbo0 = turbo_objective(random_seeds[n_seed])

        loss_turbo.append(loss_turbo0)
        mspe_turbo.append(mspe_turbo0)
        fscore_turbo.append(fscore_turbo0)

    with open('turbo_' + str(n_seed) + '.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([loss_turbo, mspe_turbo, fscore_turbo], f)
        # pickle.dump([res_loss, res_mspe, res_fscore, res_time], f)

if __name__=='__main__':
    select_seed = int(sys.argv[1])
    main_turbo(n_seed=select_seed)
