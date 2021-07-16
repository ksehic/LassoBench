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
synt_bench = LassoBench.SyntheticBenchmark(pick_bench='synt_simple3')
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
real_bench_mf = LassoBench.RealBenchmark(pick_data='rcv1', mf_opt='multi_source_bench')
d = real_bench_mf.n_features
random_config = np.random.uniform(low=-1.0, high=1.0, size=(d,))
fidelity_pick = 0
loss = real_bench_mf.fidelity_evaluate(random_config, index_fidelity=fidelity_pick)
```
## List of synthetic benchmarks
| Name          | Dimensionality | Axis-aligned Subspace |
| :---          |     :---:      |          ---:         |
| synt_simple3  | 60    | 3 |
| synt_simple6  | 60    | 6|
| synt_medium5  | 100   | 5 |
| synt_medium10 | 100   | 10 |
| synt_high15   | 300   | 15 |
| synt_high30   | 300   | 30 |
| synt_hard30   | 1000   | 50 |
| synt_hard30   | 1000   | 100 |
## List of real world benchmarks
| Name         | Dimensionality | Approx. Axis-aligned Subspace |
| :---         |     :---:      |          ---:         |
| diabetes   | 10     | 1 |
| breast_cancer     | 30  | 8|
| leukemia| 7 129     | 72 |
| rcv1     | 19 960 | 1 542 |
| news20  | 632 983 | 1 920 |
## Cite

If you use this code, please cite: TBD
