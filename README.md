# LASSOBench

LASSOBench is a library for high-dimensional hyperparameter optimization benchmark based on Weighted LASSO regression.

**Note:** LASSOBench is under active construction. Follow for more benchmarks soon. 

## Install and work with the development version

From a console or terminal clone the repository and install LASSOBench:

::

    git clone https://github.com/ksehic/LASSOBench.git
    cd celer/
    pip install -e .
    
## Simple synthetic bench code
```python
import LASSOBench
synt_bench = LASSOBench.Synt_bench()
d = 1280
random_config = np.random.uniform(low=-1.0, high=1.0, size=(d,))
loss = synt_bench.evaluate(random_config)
mspe, fscore = synt_bench.test(random_config)
```
    
## Cite

If you use this code, please cite: TBD
