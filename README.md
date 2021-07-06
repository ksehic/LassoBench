# LASSOBench

LASSOBench is a library for high-dimensional hyperparameter optimization benchmark based on Weighted LASSO regression.

**Note:** LASSOBench is under active construction. Follow for more benchmarks soon.

## Install and work with the development version

From a console or terminal clone the repository and install LASSOBench:

::

    git clone https://github.com/ksehic/LASSOBench.git
    cd LASSOBench/
    pip install -e .

## Simple synthetic bench code
```python
import LASSOBench
synt_bench = LASSOBench.Synt_bench(pick_bench='synt_high_noise_bench')
d = synt_bench.n_features
random_config = np.random.uniform(low=-1.0, high=1.0, size=(d,))
loss = synt_bench.evaluate(random_config)
```
## Real-world bench code
```python
import LASSOBench
real_bench = LASSOBench.Realworld_bench(pick_data='rcv1')
d = synt_bench.n_features
random_config = np.random.uniform(low=-1.0, high=1.0, size=(d,))
loss = real_bench.evaluate(random_config)
```
## Multi-source bench code
```python
import LASSOBench
real_bench = LASSOBench.Realworld_bench(pick_data='rcv1', mf_opt='multi_source_bench')
d = synt_bench.n_features
random_config = np.random.uniform(low=-1.0, high=1.0, size=(d,))
fidelity_pick = 0
loss = synt_bench.fidelity_evaluate(random_config, index_fidelity=fidelity_pick)
```
## List of syntentic benchmarks
| Name         | Dimensionality | Axis-aligned Subspace |
| :---         |     :---:      |          ---:         |
| synt_low_eff_bench   | 256     | 8 |
| synt_high_eff_bench     | 256  | 20|
| synt_high_noise_bench| 256     | 8 |
| synt_high_corr_bench     | 256 | 8 |
| synt_hard_bench  | 1280 | 10 |
## List of real-world benchmarks
| Name         | Dimensionality | Approx. Axis-aligned Subspace |
| :---         |     :---:      |          ---:         |
| diabetes   | 10     | 1 |
| breast_cancer     | 30  | 8|
| leukemia| 7 129     | 72 |
| rcv1     | 19 960 | 1 542 |
| news20  | 632 983 | 1 920 |
## Cite

If you use this code, please cite: TBD
