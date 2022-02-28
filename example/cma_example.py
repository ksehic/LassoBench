#!/usr/bin/python
"""
==================================================
LassoBench
High-Dimensional Hyperparameter
Optimization Benchmark
Contact: kenan.sehic@cs.lth.se

Example: CMA-ES on synthetic benchmark
You need to install CMA-ES via python -m pip install cma
Run in terminal: python cma_example.py synt_hard False 5000 20 0.1 1
=================================================
"""

import numpy as np
import LassoBench
import cma
import sys
import time
import shutil
import pickle


def main_cma(data_pick, noise, n_eval, n_doe, sigma, n_seed):
    """Running CMA-ES for synthetic benchmark

    Args:
        data_pick (str): select synthetic benchmark
        noise (boolean): add noise or noiseless
        n_eval (int): number of evaluations
        n_doe (int): population size
        sigma (int): standard deviation of sampling
        n_seed (int): seed number

    """

    synt_bench = LassoBench.SyntheticBenchmark(pick_bench=data_pick, noise=eval(noise))
    d = synt_bench.n_features

    def run_cma(n_seed):
        """Running main optimization algorithm for CMA-ES
        You need to install CMA-ES via python -m pip install cma

        Args:
            n_seed (int): seed number

        Returns:
            eval_res : MSE (Validation loss) on training dataset
            mspe : MSE on test dataset
            fscore : Fscore
            time_last : runtime performance
        """

        opts = cma.CMAOptions()
        opts.set('maxfevals', n_eval)
        opts.set('maxiter', 2000000)
        opts.set('popsize', n_doe)
        opts.set('seed', n_seed)
        opts.set('bounds', [-1, 1])
        opts.set('verb_filenameprefix', data_pick + '_outcmaes_' + str(n_seed) + '/')

        es = cma.CMAEvolutionStrategy(np.zeros((d,)), sigma, opts)
        es.optimize(synt_bench.evaluate)

        res = es.logger.load()

        eval_res = res.data['f']
        time_last = es.time_last_displayed

        x_best = res.xrecent
        mspe = np.empty((es.result.iterations,))
        fscore = np.empty((es.result.iterations,))

        for i in range(es.result.iterations):
            mspe[i], fscore[i] = synt_bench.test(x_best[i, 5:])

        shutil.rmtree(data_pick + '_outcmaes_' + str(n_seed))

        return eval_res, mspe, fscore, time_last

    n_repeat = 3000
    random_seeds = np.random.randint(200000000, size=n_repeat)

    # run cma-es
    loss_cma, mspe_cma, fscore_cma, time_cma = run_cma(random_seeds[n_seed])

    if eval(noise) is True:
        noise_pick = 'noise'
        with open('cma_init_fscore_' + noise_pick + data_pick + str(n_seed) + '.pkl', 'wb') as f:
            pickle.dump([loss_cma, mspe_cma, fscore_cma, time_cma], f)
    else:
        noise_pick = 'no'
        with open('cma_init_fscore_' + noise_pick + data_pick + str(n_seed) +'.pkl', 'wb') as f:
            pickle.dump([loss_cma, mspe_cma, fscore_cma, time_cma], f)


if __name__ == '__main__':
    bench_name = str(sys.argv[1])
    noise = sys.argv[2]
    n_total = int(sys.argv[3])
    n_doe = int(sys.argv[4])
    sigma = float(sys.argv[5])
    n_seed = int(sys.argv[6])

    main_cma(data_pick=bench_name, noise=noise, n_eval=n_total, n_doe=n_doe, sigma=sigma, n_seed=n_seed)
