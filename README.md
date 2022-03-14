# LassoBench

LassoBench is a library for high-dimensional hyperparameter optimization benchmarks based on Weighted Lasso regression.

## Install the development version

From a console or terminal clone the repository and install LassoBench:

::

    git clone https://github.com/ksehic/LassoBench.git
    cd LassoBench/
    pip install -e .

## Overview
The objective is to optimize the multi-dimensional hyperparameter space that balances
the least-squares estimation and the penalty term that promotes the sparsity.

The search space bounds are defined between [-1, 1].

LassoBench comes with two classes `SyntheticBenchmark` and `RealBenchmark`. While `RealBenchmark` is
based on real-world applications found in medicine and finance, `SyntheticBenchmark` covers synthetic well-defined conditions. The user can select one of the predefined synthetic benchmarks or create a different bechmark. For the synthetic benchmarks, the default condition for the noise level is noiseless (noise=False).

Each benchmark comes with `.evaluate` that is used to evaluate the objective function, `.test` that provides the post-processing metrics (such as MSE on the test data and the F-score for synt benchs) and the argument `mf_opt` to define the multi-fidelity framework that is evaluated via `.fidelity_evaluate`.

Simple experiments are provided in `example/example.py` where you can ran random search on different benchmarks.

## LassoBench baselines
LassoBench comes with the baselines commonly found in the Lasso community listed in the table that should be used for the comparison. LassoCV and AdaptiveLassoCV are the Lasso-based baselines where a single hyperparameter is optimized via grid search. The user can change the number of points in the grid following the provided documentation. The default value is 100 points. The implementation of the Lasso-based baselines is derived from Celer. Sparse-HO is a sparse hyperparameter optimizer based on coordinate descent. It can be easily applied to the 1D Lasso problem as well as to the Weighted Lasso problem. The user can change the number of steps, gradient solver, and similar following the provided documentation.

| Baseline          | Status | Description | Command |
| :---          |     :---:      |          ---:         |        ---:         |
| LassoCV | Included    | Standard 1D sparse regression approach | `.run_LASSOCV` |
| AdaptiveLassoCV  | To be implemented soon until then refer to the branch "adaptivelassocv" in https://github.com/mathurinm/celer | Iterative LassoCV approach | NA |
| Sparse-HO   | Included   | Sparse multi-dimensional optimizer | `.run_sparseho` |

## HPO Methods
In the folder `~/example`, the user can learn how to use `LassoBench` with some well-known HPO algorithms for high-dimensional problems `hesbo_example.py`, `cma_example.py`, `turbo_example.py` and `alebo_example.py`. Please refer to the docstrings and the table for more details.

   .
    ├── ...
    ├── example                    # Examples how to use LassoBench for HDBO algorithms
    │   ├── alebo_example.py       # ALEBO applied on synt bench
    │   ├── cma__example.py       # CMA-ES applied on synt bench
    │   ├── turbo_example.py       # TuRBO applied on synt bench
    │   ├── example.py             # Simple cases how to run with synt, real, and multifidelity benchs
    │   ├── hesbo_example.py        # HeSBO applied on synt and real bench
    │   ├── hesbo_lib.pu            # HeSBO library
    │
    └── ...

| HPO Methods          | Install | Description | File |
| :---          |     :---:      |          ---:         |        ---:         |
| HeSBO | Check HeSBO lib    | Bayesian Optimization with dimensionality reduction | `hesbo_example.py` |
| ALEBO  | Install prerequirements | Bayesian Optimization with dimensionality reduction | `alebo_example.py` |
| CMA-ES   | `python -m pip install cma`   | Evolutionary Strategy  | `cma_example.py` |
| TuRBO   | Follow https://github.com/uber-research/TuRBO | Local Bayesian Optimization | `turbo_example.py`|

## License

LassoBench is distributed under the MIT license. More information on the license can be found [here](https://github.com/ksehic/LassoBench/blob/main/LICENSE)

## Simple noiseless [synthetic](#list-of-synthetic-benchmarks) bench code
```python
import numpy as np
import LassoBench
synt_bench = LassoBench.SyntheticBenchmark(pick_bench='synt_simple')
d = synt_bench.n_features
random_config = np.random.uniform(low=-1.0, high=1.0, size=(d,))
loss = synt_bench.evaluate(random_config)
```
## Simple noisy [synthetic](#list-of-synthetic-benchmarks) bench code
```python
import numpy as np
import LassoBench
synt_bench = LassoBench.SyntheticBenchmark(pick_bench='synt_simple', noise=True)
d = synt_bench.n_features
random_config = np.random.uniform(low=-1.0, high=1.0, size=(d,))
loss = synt_bench.evaluate(random_config)
```
## [Real-world](#list-of-real-world-benchmarks) bench code
```python
import numpy as np
import LassoBench
real_bench = LassoBench.RealBenchmark(pick_data='rcv1')
d = real_bench.n_features
random_config = np.random.uniform(low=-1.0, high=1.0, size=(d,))
loss = real_bench.evaluate(random_config)
```
## Multi-information source bench code
```python
import numpy as np
import LassoBench
real_bench_mf = LassoBench.RealBenchmark(pick_data='rcv1', mf_opt='discrete_fidelity')
d = real_bench_mf.n_features
random_config = np.random.uniform(low=-1.0, high=1.0, size=(d,))
fidelity_pick = 0
loss = real_bench_mf.fidelity_evaluate(random_config, index_fidelity=fidelity_pick)
```
## List of synthetic benchmarks
| Name          | Dimensionality | Axis-aligned Subspace |
| :---          |     :---:      |          ---:         |
| synt_simple  | 60    | 3 |
| synt_medium  | 100   | 5 |
| synt_high   | 300   | 15 |
| synt_hard   | 1000   | 50 |
## List of real world benchmarks
| Name         | Dimensionality | Approx. Axis-aligned Subspace |
| :---         |     :---:      |          ---:         |
| breast_cancer | 10 | 3 |
| diabetes | 8 | 5 |
| leukemia | 7 129 | 22 |
| dna | 180 | 43 |
| rcv1 | 19 959 | 75 |

## Cite

If you use this code, please cite:

```

Šehić Kenan, Gramfort Alexandre, Salmon Joseph and Nardi Luigi, "LassoBench: A High-Dimensional Hyperparameter Optimization Benchmark Suite for Lasso", under review, 2022.

```
