# LassoBench

LassoBench is a library for high-dimensional hyperparameter optimization benchmark based on Weighted LASSO regression.

**Note:** LassoBench is under active construction. Follow for more benchmarks soon.

## Install and work with the development version

From a console or terminal clone the repository and install LassoBench:

::

    git clone https://github.com/ksehic/LassoBench.git
    cd LassoBench/
    pip install -e .

## Overview
The objective is to optimize the multi-dimensional hyperparameter that balances
the least-squares estimation and the penalty term that promotes the sparsity.

The ambient space bounds are defined between [-1, 1].

LassoBench comes with two classes SyntheticBenchmark and RealBenchmark. While RealBenchmark is
based on real-world applications found in medicine and finance, SyntheticBenchmark covers synthetic well-defined conditions.

The results are compared with the baselines LassoCV, AdaptiveLassoCV (to be implemented soon) and Sparse-HO.

Please refer to the reference for more details.

    .
    ├── ...
    ├── example                    # Examples how to use LassoBench for HDBO algorithms
    │   ├── alebo_example.py       # ALEBO applied on synt bench
    │   ├── example.py             # Simple cases how to run with synt, real and multifidelity benchs
    │   ├── hesbo_example.py        # HesBO applied on synt bench
    │   ├── hesbo_lib.pu            # HesBO library
    │
    └── ...

## Simple [synthetic](#list-of-synthetic-benchmarks) bench code
```python
import numpy as np
import LassoBench
synt_bench = LassoBench.SyntheticBenchmark(pick_bench='synt_simple')
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
## Multi-source bench code
```python
import numpy as np
import LassoBench
real_bench_mf = LassoBench.RealBenchmark(pick_data='rcv1', mf_opt='multi_discrete')
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
| breast_cancer | 10 | 683 | 3 |
| diabetes | 8 | 768 | 5 |
| leukemia | 7 129 | 38 | 22 |
| dna | 180 | 2 000 | 43 |
| rcv1 | 19 959 | 20 242 | 75 |

## Cite

If you use this code, please cite: TBD
