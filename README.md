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
The objective is to optimize the multi-dimensional hyperparameter $\lambda$ that balances
the least-squares estimation and the penalty term that promotes the sparsity.

The ambient space bounds are defined between $\lambda\in[-1, 1]$.

LassoBench comes with two classes SyntheticBenchmark and RealBenchmark. While RealBenchmark is
based on real world applications found in medicine and finance, SyntheticBenchmark covers synthentic well-defined conditions.

Please refer the reference for more details.

.
├── ...
├── example                    # Examples how to use LassoBench for HDBO algorithms
│   ├── alebo_example.py       # ALEBO applied on synt bench
│   ├── example.py             # Simple cases with synt, real and multifidelity benchs
│   ├── hesbo_example.py        # HesBO applied on synt bench
│   ├── hesbo_lib.pu            # HesBO library
│
└── ...

## Simple [synthetic](#list-of-synthetic-benchmarks) bench code
```python
import numpy as np
import LassoBench
synt_bench = LassoBench.SyntheticBenchmark(pick_bench='synt_high_noise_bench')
d = synt_bench.n_features
random_config = np.random.uniform(low=-1.0, high=1.0, size=(d,))
loss = synt_bench.evaluate(random_config)
```
## ## [Real-world](#list-of-real-world-benchmarks) bench code
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
| Name         | Dimensionality | Axis-aligned Subspace |
| :---         |     :---:      |          ---:         |
| synt_low_eff_bench   | 256     | 8 |
| synt_high_eff_bench     | 256  | 20|
| synt_high_noise_bench| 256     | 8 |
| synt_high_corr_bench     | 256 | 8 |
| synt_hard_bench  | 1280 | 10 |
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
