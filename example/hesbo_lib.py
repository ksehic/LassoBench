#!/usr/bin/python
"""
==================================================
HesBO - A Framework for Bayesian Optimization in Embedded Subspaces

LINK: https://github.com/aminnayebi/HesBO

@inproceedings{HeSBO19,
  author    = {Alex Munteanu and
               Amin Nayebi and
			   Matthias Poloczek},
  title     = {A Framework for Bayesian Optimization in Embedded Subspaces},
  booktitle = {Proceedings of the 36th International Conference on Machine Learning, {(ICML)}},
  year      = {2019},
  note={Accepted for publication. The code is available at https://github.com/aminnayebi/HesBO.}
}
=================================================
"""

import numpy as np
from scipy.stats import norm
import timeit
from pyDOE import lhs
import GPy

def EI(D_size, f_max, mu, var):
    """
    :param D_size: number of points for which EI function will be calculated
    :param f_max: the best value found for the test function so far
    :param mu: a vector of predicted values for mean of the test function
        corresponding to the points
    :param var: a vector of predicted values for variance of the test function
        corresponding to the points
    :return: a vector of EI values of the points
    """
    ei=np.zeros((D_size,1))
    std_dev=np.sqrt(var)
    for i in range(D_size):
        if var[i]!=0:
            z= (mu[i] - f_max) / std_dev[i]
            ei[i]= (mu[i]-f_max) * norm.cdf(z) + std_dev[i] * norm.pdf(z)
    return ei

def dim_sampling(low_dim, X, bx_size):
    if len(X.shape)==1:
        X=X.reshape((1, X.shape[0]))
    n=X.shape[0]
    high_dim=X.shape[1]
    low_obs=np.zeros((n,low_dim))
    high_to_low=np.zeros(high_dim,dtype=int)
    sign=np.random.choice([-1,1],high_dim)
    for i in range(high_dim):
        high_to_low[i]=np.random.choice(range(low_dim))
        low_obs[:,high_to_low[i]]=X[:,i]*sign[i]+ low_obs[:,high_to_low[i]]

    for i in range(n):
        for j in range(low_dim):
            if low_obs[i][j] > bx_size: low_obs[i][j] = bx_size
            elif low_obs[i][j] < -bx_size: low_obs[i][j] = -bx_size
    return low_obs, high_to_low, sign


def back_projection(low_obs, high_to_low, sign, bx_size):
    if len(low_obs.shape)==1:
        low_obs=low_obs.reshape((1, low_obs.shape[0]))
    n=low_obs.shape[0]
    high_dim=high_to_low.shape[0]
    low_dim=low_obs.shape[1]
    high_obs=np.zeros((n,high_dim))
    scale=1
    for i in range(high_dim):
        high_obs[:, i]=sign[i]*low_obs[:,high_to_low[i]]*scale
    for i in range(n):
        for j in range(high_dim):
            if high_obs[i][j] > bx_size: high_obs[i][j] = bx_size
            elif high_obs[i][j] < -bx_size: high_obs[i][j] = -bx_size
    return high_obs


def RunMain(low_dim=2, high_dim=25, initial_n=20, total_itr=100, test_func=None,
            s=None, active_var=None, ARD=False, variance=1., length_scale=None, box_size=None,
            high_to_low=None, sign=None, hyper_opt_interval=20, n_seed=42):
    """

    :param high_dim: the dimension of high dimensional search space
    :param low_dim: The effective dimension of the algorithm.
    :param initial_n: the number of initial points
    :param total_itr: the number of iterations of algorithm. The total
        number of test function evaluations is initial_n + total_itr
    :param test_func: test function
    :param s: initial points
    :param active_var: a vector with the size of greater or equal to
        the number of active variables of test function. The values of
        vector are integers less than high_dim value.
    :param ARD: if TRUE, kernel is isomorphic
    :param variance: signal variance of the kernel
    :param length_scale: length scale values of the kernel
    :param box_size: this variable indicates the search space [-box_size, box_size]^d
    :param high_to_low: a vector with D elements. each element can have a value from {0,..,d-1}
    :param sign: a vector with D elements. each element is either +1 or -1.
    :param hyper_opt_interval: the number of iterations between two consecutive
        hyper parameters optimizations
    :return: a tuple of best values of each iteration, all observed points, and
        corresponding test function values of observed points
    """

    np.random.seed(n_seed)

    if active_var is None:
        active_var= np.arange(high_dim)
    if box_size is None:
        box_size=1
    if high_to_low is None:
        high_to_low=np.random.choice(range(low_dim), high_dim)
    if sign is None:
        sign = np.random.choice([-1, 1], high_dim)

    best_results = np.zeros([1, total_itr + initial_n])
    elapsed=np.zeros([1, total_itr + initial_n])

    start = timeit.default_timer()

    # Creating the initial points. The shape of s is nxD
    if s is None:
        s=lhs(low_dim, initial_n) * 2 * box_size - box_size
    f_s, config, time_init = test_func(back_projection(s,high_to_low,sign,box_size))
    for i in range(initial_n):
        best_results[0,i]=np.max(f_s[0:i+1])

    # Building and fitting a new GP model
    kern = GPy.kern.Matern52(input_dim=low_dim, ARD=ARD, variance=variance, lengthscale=length_scale)
    m = GPy.models.GPRegression(s, f_s, kernel=kern)
    m.likelihood.variance = 1e-3

    # Main loop
    for i in range(total_itr):

        # Updating GP model
        m.set_XY(s, f_s)
        if (i+initial_n<=25 and i % 5 == 0) or (i+initial_n>25 and i % hyper_opt_interval == 0):
            m.optimize()

        # Maximizing acquisition function
        D = lhs(low_dim, 2000) * 2 * box_size - box_size
        mu, var = m.predict(D)
        ei_d = EI(len(D), max(f_s), mu, var)
        index = np.argmax(ei_d)

        # Adding the new point to our sample
        s = np.append(s, [D[index]], axis=0)
        new_high_point=back_projection(D[index],high_to_low,sign,box_size)
        run_f, new_config, _ = test_func(new_high_point)
        f_s = np.append(f_s, run_f, axis=0)
        config = np.append(config, new_config, axis=0)

        stop = timeit.default_timer()
        best_results[0, i + initial_n] = np.max(f_s)
        elapsed[0, i + initial_n]=stop-start

    elapsed[0, :initial_n] = time_init - start
    high_s = back_projection(s,high_to_low,sign,box_size)

    return best_results, elapsed, s, f_s, high_s, config
