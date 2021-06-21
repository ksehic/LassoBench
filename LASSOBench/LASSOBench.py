#!/usr/bin/python
"""
==================================================
LASSOBench
High-Dimensional Hyperparameter
Optimization Benchmark
Contact: kenan.sehic@cs.lth.se
=================================================
"""
import numpy as np
from celer import Lasso, LassoCV

from sparse_ho.models import WeightedLasso
from sparse_ho.criterion import HeldOutMSE, CrossVal
from sparse_ho.utils import Monitor
from sparse_ho import Implicit, ImplicitForward, grad_search
from sparse_ho.optimizers import LineSearch, GradientDescent, Adam

from celer.datasets import make_correlated_data

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score

from libsvmdata import fetch_libsvm

from joblib import Parallel, delayed
import multiprocessing

from LASSOBench.hesbo_lib import RunMain
import timeit


class Synt_bench():
    def __init__(self, n_features=1280, n_samples=640,
                 snr_level=1, corr_level=0.6, n_nonzeros=10,
                 w_true=None, n_splits=5, test_size=0.15,
                 tol_level=1e-4, eps_support=1e-6, seed=42):
        """
        Synthetic Benchmark that is used to test a HPO algorithm
        on different conditions. It is based on the cross-validation critetion.
        Args:
            input_config: numpy array sampled within [-1, 1] with d number of elements
            n_features: the size of search space d>0 (i.e., the number of features)
            n_samples: the number of samples in a dataset
            snr_level: the level of noise with SNR=1 being very noisy and SNR=10 is almost noiseless.
            corr_level: the level of correlation withint features
            n_nonzeros: the number of nonzero elements in true reg coef betas
                        that can reduce or increase the sparsity of the solution
            w_true: the predefined reg coef betas numpy array with elemenets equal to n_features
            n_splits: the number of data splits for cross-validation
            test_size: the percentage of test data
            eps_support: the support threshold
            seed: the seed number
        Return:
            evaluate: val_loss (the cross-validation loss for evaluate)
            test: mspe_div (the mean-squared prediction error divided by the oracle error)
                  fscore (the F-measure for support recovery)
            fidelity_evaluate: val_loss for each fidelity (use predefined fidelitis or use tol level to generate a fidelity)
            run_hesbo: loss, mspe, fscore and elapsed for HesBO (high-dimensional BO algorithm)
        """

        self.tol_level = tol_level
        self.n_features = n_features
        self.n_splits = n_splits

        X, y, self.w_true = make_correlated_data(
            n_samples=n_samples, n_features=n_features,
            corr=corr_level, w_true=w_true,
            snr=snr_level, density=n_nonzeros/n_features,
            random_state=seed)

        # split train and test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed)

        self.kf = KFold(shuffle=True, n_splits=self.n_splits, random_state=seed)

        self.alpha_max = np.max(np.abs(
            self.X_train.T @ self.y_train)) / len(self.y_train)
        self.alpha_min = self.alpha_max / 1e2

        self.log_alpha_min = np.log(self.alpha_min)
        self.log_alpha_max = np.log(self.alpha_max)

        self.eps_support = eps_support

        self.coef_true_support = np.abs(self.w_true) > self.eps_support
        self.mspe_oracle = mean_squared_error(
            self.X_test @ self.w_true, self.y_test)

    def scale_domain(self, x):
        # Scaling the domain
        x_copy = np.copy(x)
        x_copy = x_copy * (
                self.log_alpha_max - self.log_alpha_min) / 2 + (
                    self.log_alpha_max + self.log_alpha_min) / 2
        return x_copy

    def evaluate(self, input_config):
        scaled_x = self.scale_domain(input_config)

        estimator = Lasso(fit_intercept=False, max_iter=100, warm_start=True)
        model = WeightedLasso(estimator=estimator)
        monitor = Monitor()
        sub_criterion = HeldOutMSE(None, None)
        criterion = CrossVal(sub_criterion, cv=self.kf)
        val_loss = criterion.get_val(model, self.X_train, self.y_train,
                                     log_alpha=scaled_x,
                                     monitor=monitor, tol=self.tol_level)

        return val_loss

    def fidelity_evaluate(self, input_config, index_fidelity=None,
                          n_fidelity=5, tol_fidelity=None):

        if tol_fidelity is None and index_fidelity is not None:
            tol_range = np.geomspace(self.tol_level, 0.2, num=n_fidelity)
            tol_budget = tol_range[index_fidelity]
        elif tol_fidelity is None and index_fidelity is not None:
            tol_budget = tol_fidelity
        else:
            raise ValueError("Please select only one; the level of tolerance or fidelity index.")

        scaled_x = self.scale_domain(input_config)

        estimator = Lasso(fit_intercept=False, max_iter=100, warm_start=True)
        model = WeightedLasso(estimator=estimator)
        monitor = Monitor()
        sub_criterion = HeldOutMSE(None, None)
        criterion = CrossVal(sub_criterion, cv=self.kf)
        val_loss = criterion.get_val(model, self.X_train, self.y_train,
                                     log_alpha=scaled_x,
                                     monitor=monitor, tol=tol_budget)

        return val_loss

    def test(self, input_config):
        scaled_x = self.scale_domain(input_config)
        estimator = Lasso(fit_intercept=False, max_iter=100, warm_start=True)
        estimator.weights = np.exp(scaled_x)
        estimator.fit(self.X_train, self.y_train)
        reg_coef = estimator.coef_
        coef_hpo_support = np.abs(estimator.coef_) > self.eps_support
        fscore = f1_score(self.coef_true_support, coef_hpo_support)
        mspe = mean_squared_error(estimator.predict(self.X_test), self.y_test)
        mspe_div = mspe/self.mspe_oracle

        return mspe_div, fscore, reg_coef

    def evaluate_hesbo(self, x):
        scaled_x = self.scale_domain(x)
        j = 0
        if len(scaled_x.shape) != 1:
            obj_value = np.empty((scaled_x.shape[0], ))
            config_all = np.empty((scaled_x.shape[0], self.n_features))
            time_stop = np.empty((scaled_x.shape[0], ))
        for i in scaled_x:
            config = np.empty((self.n_features,))
            for z in range(self.n_features):
                config[z] = i[z]

            estimator = Lasso(fit_intercept=False, max_iter=100, warm_start=True)
            model = WeightedLasso(estimator=estimator)
            monitor = Monitor()
            sub_criterion = HeldOutMSE(None, None)
            criterion = CrossVal(sub_criterion, cv=self.kf)

            if len(scaled_x.shape) == 1:
                obj_value = criterion.get_val(model, self.X_train, self.y_train,
                                              log_alpha=config,
                                              monitor=monitor, tol=self.tol_level)
                obj_value = np.array([obj_value])
                obj_value = obj_value.reshape(1, 1)
                config_all = config
                time_stop = timeit.default_timer()
            else:
                obj_value[j] = criterion.get_val(model, self.X_train, self.y_train,
                                log_alpha=config,
                                monitor=monitor, tol=self.tol_level)
                time_stop[j] = timeit.default_timer()
                config_all[j, :] = config
                j = j + 1

        if len(scaled_x.shape) != 1:
            obj_value = obj_value.reshape(scaled_x.shape[0], 1)

        return -obj_value, config_all, time_stop

    def run_hesbo(self, eff_dim, n_doe, n_total, ARD=True, n_repeat=0, n_seed=42, n_jobs=1):

        estimator = Lasso(fit_intercept=False, max_iter=100, warm_start=True)
        n_total = n_total + n_doe

        if n_jobs > 1:

            def run_parallel_hesbo(low_dim, high_dim, initial_n, total_itr, test_func, ARD):
                def parallel_hesbo(n_seed):
                    _, elapsed0, _, loss0, _, config0 = RunMain(low_dim=low_dim, high_dim=high_dim, initial_n=initial_n,
                                                                total_itr=total_itr, test_func=test_func, ARD=ARD,
                                                                n_seed=n_seed)
                    return elapsed0, loss0, config0
                return parallel_hesbo

            hesbo_objective = run_parallel_hesbo(low_dim=eff_dim, high_dim=self.n_features, initial_n=n_doe,
                                                 total_itr=n_total - n_doe, test_func=self.evaluate_hesbo, ARD=ARD)
            random_seeds = np.random.randint(200000000, size=n_repeat)
            # num_cores = multiprocessing.cpu_count()
            par_res = Parallel(n_jobs=n_jobs)(
                delayed(hesbo_objective)(n_seed) for n_seed in random_seeds)

            loss = np.empty((n_total, n_repeat))
            elapsed = np.empty((n_total, n_repeat))
            mspe_hesbo = np.empty((n_total, n_repeat))
            fscore = np.empty((n_total, n_repeat))
            config_all = np.empty((n_total, self.n_features, n_repeat))

            for i in range(n_repeat):
                loss[:, i] = np.squeeze(par_res[i][1])
                elapsed[:, i] = np.squeeze(par_res[i][0])
                config_par = par_res[i][2]
                for j in range(n_total):
                    estimator.weights = np.exp(config_par[j, :])
                    config_all[j, :, i] = np.exp(config_par[j, :])
                    estimator.fit(self.X_train, self.y_train)
                    mspe_hesbo[j, i] = mean_squared_error(estimator.predict(self.X_test), self.y_test)
                    coef_hesbo_support = np.abs(estimator.coef_) > self.eps_support
                    fscore[j, i] = f1_score(self.coef_true_support, coef_hesbo_support)
            mspe_hesbo = mspe_hesbo / self.mspe_oracle
        else:
            if n_repeat > 1:
                random_seeds = np.random.randint(200000000, size=n_repeat)
                loss = np.empty((n_total, n_repeat))
                elapsed = np.empty((n_total, n_repeat))
                mspe_hesbo = np.empty((n_total, n_repeat))
                fscore = np.empty((n_total, n_repeat))
                config_all = np.empty((n_total, self.n_features, n_repeat))

                for i in range(n_repeat):
                    _, elapsed0, _, loss0, _, config0 = RunMain(low_dim=eff_dim, high_dim=self.n_features, initial_n=n_doe,
                                                                total_itr=n_total - n_doe, test_func=self.evaluate_hesbo, ARD=ARD,
                                                                n_seed=random_seeds[i])
                    loss[:, i] = loss0[:, 0]
                    elapsed[:, i] = elapsed0[0, :]
                    for j in range(n_total):
                        estimator.weights = np.exp(config0[j, :])
                        config_all[j, :, i] = np.exp(config0[j, :])
                        estimator.fit(self.X_train, self.y_train)
                        mspe_hesbo[j, i] = mean_squared_error(estimator.predict(self.X_test), self.y_test)
                        coef_hesbo_support = np.abs(estimator.coef_) > self.eps_support
                        fscore[j, i] = f1_score(self.coef_true_support, coef_hesbo_support)
                mspe_hesbo = mspe_hesbo / self.mspe_oracle
            else:
                _, elapsed, _, loss, _, config = RunMain(low_dim=eff_dim, high_dim=self.n_features, initial_n=n_doe, total_itr=n_total - n_doe, test_func=self.evaluate_hesbo, ARD=ARD, n_seed=n_seed)
                mspe_hesbo = np.empty((n_total,))
                fscore = np.empty((n_total,))
                config_all = np.empty((n_total, self.n_features))
                for i in range(n_total):
                    estimator.weights = np.exp(config[i, :])
                    config_all[i, :] = np.exp(config[i, :])
                    estimator.fit(self.X_train, self.y_train)
                    mspe_hesbo[i] = mean_squared_error(estimator.predict(self.X_test), self.y_test)
                    coef_hesbo_support = np.abs(estimator.coef_) > self.eps_support
                    fscore[i] = f1_score(self.coef_true_support, coef_hesbo_support)
                mspe_hesbo = mspe_hesbo / self.mspe_oracle

        return -loss, mspe_hesbo, fscore, config_all, elapsed

    def run_LASSOCV(self, n_alphas=100):

        # default number of alphas
        alphas = np.geomspace(self.alpha_max, self.alpha_min, n_alphas)

        lasso_params = dict(fit_intercept=False, tol=self.tol_level,
                            normalize=False, cv=self.kf, n_jobs=self.n_splits)

        # run LassoCV celer
        t0 = timeit.default_timer()
        model_lcv = LassoCV(alphas=alphas, **lasso_params)
        model_lcv.fit(self.X_train, self.y_train)
        t1 = timeit.default_timer()
        elapsed = t1 - t0

        min_lcv = np.where(model_lcv.mse_path_ == np.min(model_lcv.mse_path_))
        loss_lcv = np.mean(model_lcv.mse_path_[min_lcv[0]])

        mspe_lcv = mean_squared_error(
            model_lcv.predict(self.X_test), self.y_test)

        coef_lcv_support = np.abs(model_lcv.coef_) > self.eps_support
        fscore = f1_score(self.coef_true_support, coef_lcv_support)

        return loss_lcv, mspe_lcv/self.mspe_oracle, fscore, alphas[min_lcv[0]], elapsed

    def run_sparseho(self, grad_solver='gd', algo_pick='imp_forw', n_steps=10, init_point=None, verbose=False):

        if init_point is not None:
            init_point_scale = self.scale_domain(init_point)

        estimator = Lasso(fit_intercept=False, max_iter=100, warm_start=True)
        model = WeightedLasso(estimator=estimator)
        sub_criterion = HeldOutMSE(None, None)
        criterion = CrossVal(sub_criterion, cv=self.kf)
        if algo_pick == 'imp_forw':
            algo = ImplicitForward()
        elif algo_pick == 'imp':
            algo = Implicit()
        else:
            raise ValueError("Undefined algo_pick.")

        monitor = Monitor()
        if grad_solver == 'gd':
            optimizer = GradientDescent(n_outer=n_steps, tol=self.tol_level,
                                        verbose=verbose, p_grad_norm=1.9)
        elif grad_solver == 'adam':
            optimizer = Adam(n_outer=n_steps, lr=0.11, verbose=verbose, tol=self.tol_level)
        elif grad_solver == 'line':
            optimizer = LineSearch(n_outer=n_steps, verbose=verbose, tol=self.tol_level)

        if init_point is None:
            grad_search(
                algo, criterion, model, optimizer, self.X_train, self.y_train,
                self.alpha_max/10*np.ones((self.X_train.shape[1],)), monitor)
        else:
            grad_search(
                algo, criterion, model, optimizer, self.X_train, self.y_train,
                init_point_scale, monitor)

        mspe = np.empty((n_steps,))
        fscore = np.empty((n_steps,))
        config_all = np.empty((n_steps, self.n_features))
        reg_coef = np.empty((n_steps, self.n_features))

        for i in range(n_steps):
            estimator.weights = monitor.alphas[i]
            config_all[i, :] = monitor.alphas[i]
            estimator.fit(self.X_train, self.y_train)
            mspe[i] = mean_squared_error(estimator.predict(self.X_test), self.y_test)
            reg_coef[i, :] = estimator.coef_
            coef_sho_support = np.abs(estimator.coef_) > self.eps_support
            fscore[i] = f1_score(self.coef_true_support, coef_sho_support)

        return monitor.objs, mspe/self.mspe_oracle, fscore, config_all, reg_coef, monitor.times



class Realworld_bench():
    def __init__(self, data_pick=None, n_splits=5, test_size=0.15,
                 tol_level=1e-4, seed=42):
        """
        Real world Benchmark that is used to test a HPO algorithm
        for datasets found in practice. It is based on the cross-validation critetion.
        Args:
            input_config: numpy array sampled within [-1, 1] with d number of elements
            data_pick: select real world dataset
            n_splits: the number of data splits for cross-validation
            test_size: the percentage of test data
            seed: the seed number
        Return:
            evaluate: val_loss (the cross-validation loss for evaluate)
            test: mspe (the mean-squared prediction error)
            fidelity_evaluate: val_loss for each fidelity (use predefined fidelitis or use tol level to generate a fidelity)
            run_hesbo: loss, mspe and elapsed for HesBO (high-dimensional BO algorithm)
        """

        self.tol_level = tol_level

        if data_pick == 'diabetes':
            X, y = fetch_libsvm('diabetes_scale')
            alpha_scale = 1e5
        elif data_pick == 'breast-cancer':
            X, y = fetch_libsvm('breast-cancer_scale')
            alpha_scale = 1e5
        elif data_pick == 'leukemia':
            X, y = fetch_libsvm(data_pick)
            alpha_scale = 1e5
        elif data_pick == 'rcv1':
            X, y = fetch_libsvm('rcv1.binary')
            alpha_scale = 1e3
        elif data_pick == 'news20':
            X, y = fetch_libsvm('news20.binary')
            alpha_scale = 1e5
        else:
            raise ValueError("Unsupported dataset %s" % data_pick)

        self.n_features = X.shape[1]

        # split train and test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed)

        self.kf = KFold(shuffle=True, n_splits=n_splits, random_state=seed)

        self.alpha_max = np.max(np.abs(
            self.X_train.T @ self.y_train)) / len(self.y_train)
        self.alpha_min = self.alpha_max / alpha_scale

        self.log_alpha_min = np.log(self.alpha_min)
        self.log_alpha_max = np.log(self.alpha_max)

    def scale_domain(self, input_config):
        # Scaling the domain for [-1, 1]
        input_config_copy = np.copy(input_config)
        old_min = -1
        old_max = 1
        scale_input = ((input_config_copy - old_min) / (old_max - old_min)) * (
            self.log_alpha_max - self.log_alpha_min)
        scale_input = scale_input + self.log_alpha_min

        return scale_input

    def evaluate(self, input_config):
        scaled_x = self.scale_domain(input_config)

        estimator = Lasso(fit_intercept=False, max_iter=100, warm_start=True)
        model = WeightedLasso(estimator=estimator)
        monitor = Monitor()
        sub_criterion = HeldOutMSE(None, None)
        criterion = CrossVal(sub_criterion, cv=self.kf)
        val_loss = criterion.get_val(model, self.X_train, self.y_train,
                                     log_alpha=scaled_x,
                                     monitor=monitor, tol=self.tol_level)

        return val_loss

    def fidelity_evaluate(self, input_config, index_fidelity=None,
                          n_fidelity=5, tol_fidelity=None):

        if tol_fidelity is None and index_fidelity is not None:
            tol_range = np.geomspace(self.tol_level, 0.2, num=n_fidelity)
            tol_budget = tol_range[index_fidelity]
        elif tol_fidelity is None and index_fidelity is not None:
            tol_budget = tol_fidelity
        else:
            raise ValueError("Please select only one; the level of tolerance or fidelity index.")

        scaled_x = self.scale_domain(input_config)

        estimator = Lasso(fit_intercept=False, max_iter=100, warm_start=True)
        model = WeightedLasso(estimator=estimator)
        monitor = Monitor()
        sub_criterion = HeldOutMSE(None, None)
        criterion = CrossVal(sub_criterion, cv=self.kf)
        val_loss = criterion.get_val(model, self.X_train, self.y_train,
                                     log_alpha=scaled_x,
                                     monitor=monitor, tol=tol_budget)

        return val_loss

    def test(self, input_config):
        scaled_x = self.scale_domain(input_config)
        estimator = Lasso(fit_intercept=False, max_iter=100, warm_start=True)
        estimator.weights = np.exp(scaled_x)
        estimator.fit(self.X_train, self.y_train)
        mspe = mean_squared_error(estimator.predict(self.X_test), self.y_test)

        return mspe

    def scale_domain_hesbo(self, x):
        # Scaling the domain for [0, 1]
        x_copy = np.copy(x)
        x_copy = x_copy * (
                self.log_alpha_max - self.log_alpha_min) / 2 + (
                    self.log_alpha_max + self.log_alpha_min) / 2
        return x_copy

    def evaluate_hesbo(self, x):
        scaled_x = self.scale_domain_hesbo(x)
        j = 0
        if len(scaled_x.shape) != 1:
            obj_value = np.empty((scaled_x.shape[0], ))
            config_all = np.empty((scaled_x.shape[0], self.n_features))
            time_stop = np.empty((scaled_x.shape[0], ))
        for i in scaled_x:
            config = np.empty((self.n_features,))
            for z in range(self.n_features):
                config[z] = i[z]

            estimator = Lasso(fit_intercept=False, max_iter=100, warm_start=True)
            model = WeightedLasso(estimator=estimator)
            monitor = Monitor()
            sub_criterion = HeldOutMSE(None, None)
            criterion = CrossVal(sub_criterion, cv=self.kf)

            if len(scaled_x.shape) == 1:
                obj_value = criterion.get_val(model, self.X_train, self.y_train,
                                              log_alpha=config,
                                              monitor=monitor, tol=self.tol_level)
                obj_value = np.array([obj_value])
                obj_value = obj_value.reshape(1, 1)
                config_all = config
                time_stop = timeit.default_timer()
            else:
                obj_value[j] = criterion.get_val(model, self.X_train, self.y_train,
                                log_alpha=config,
                                monitor=monitor, tol=self.tol_level)
                time_stop[j] = timeit.default_timer()
                config_all[j, :] = config
                j = j + 1

        if len(scaled_x.shape) != 1:
            obj_value = obj_value.reshape(scaled_x.shape[0], 1)

        return -obj_value, config_all, time_stop

    def run_hesbo(self, eff_dim, n_doe, n_total, ARD=True, n_repeat=0, n_seed=42, n_jobs=1):

        estimator = Lasso(fit_intercept=False, max_iter=100, warm_start=True)

        if n_jobs > 1:

            def run_parallel_hesbo(low_dim, high_dim, initial_n, total_itr, test_func, ARD):
                def parallel_hesbo(n_seed):
                    _, elapsed0, _, loss0, _, config0 = RunMain(low_dim=low_dim, high_dim=high_dim, initial_n=initial_n,
                                                                total_itr=total_itr, test_func=test_func, ARD=ARD,
                                                                n_seed=n_seed)
                    return elapsed0, loss0, config0
                return parallel_hesbo

            hesbo_objective = run_parallel_hesbo(low_dim=eff_dim, high_dim=self.n_features, initial_n=n_doe,
                                                 total_itr=n_total - n_doe, test_func=self.evaluate_hesbo, ARD=ARD)
            random_seeds = np.random.randint(200000000, size=n_repeat)
            # num_cores = multiprocessing.cpu_count()
            par_res = Parallel(n_jobs=n_jobs)(
                delayed(hesbo_objective)(n_seed) for n_seed in random_seeds)

            loss = np.empty((n_total, n_repeat))
            elapsed = np.empty((n_total, n_repeat))
            mspe_hesbo = np.empty((n_total, n_repeat))

            for i in range(n_repeat):
                loss[:, i] = np.squeeze(par_res[i][1])
                elapsed[:, i] = np.squeeze(par_res[i][0])
                config_par = par_res[i][2]
                for j in range(n_total):
                    estimator.weights = np.exp(config_par[j, :])
                    estimator.fit(self.X_train, self.y_train)
                    mspe_hesbo[j, i] = mean_squared_error(estimator.predict(self.X_test), self.y_test)
        else:
            if n_repeat > 1:
                random_seeds = np.random.randint(200000000, size=n_repeat)
                loss = np.empty((n_total, n_repeat))
                elapsed = np.empty((n_total, n_repeat))
                mspe_hesbo = np.empty((n_total, n_repeat))

                for i in range(n_repeat):
                    _, elapsed0, _, loss0, _, config0 = RunMain(low_dim=eff_dim, high_dim=self.n_features, initial_n=n_doe,
                                                                total_itr=n_total - n_doe, test_func=self.evaluate_hesbo, ARD=ARD,
                                                                n_seed=random_seeds[i])
                    loss[:, i] = loss0[:, 0]
                    elapsed[:, i] = elapsed0[0, :]
                    for j in range(n_total):
                        estimator.weights = np.exp(config0[j, :])
                        estimator.fit(self.X_train, self.y_train)
                        mspe_hesbo[j, i] = mean_squared_error(estimator.predict(self.X_test), self.y_test)
            else:
                _, elapsed, _, loss, _, config = RunMain(low_dim=eff_dim, high_dim=self.n_features, initial_n=n_doe, total_itr=n_total - n_doe, test_func=self.evaluate_hesbo, ARD=ARD, n_seed=n_seed)
                mspe_hesbo = np.empty((n_total,))
                for i in range(n_total):
                    estimator.weights = np.exp(config[i, :])
                    estimator.fit(self.X_train, self.y_train)
                    mspe_hesbo[i] = mean_squared_error(estimator.predict(self.X_test), self.y_test)

        return loss, mspe_hesbo, elapsed

    def run_LASSOCV(self, n_alphas=100):

        # default number of alphas
        alphas = np.geomspace(self.alpha_max, self.alpha_min, n_alphas)

        lasso_params = dict(fit_intercept=False, tol=self.tol_level,
                            normalize=False, cv=self.kf, n_jobs=self.n_splits)

        # run LassoCV celer
        t0 = timeit.default_timer()
        model_lcv = LassoCV(alphas=alphas, **lasso_params)
        model_lcv.fit(self.X_train, self.y_train)
        t1 = timeit.default_timer()
        elapsed = t1 - t0

        min_lcv = np.where(model_lcv.mse_path_ == np.min(model_lcv.mse_path_))
        loss_lcv = np.mean(model_lcv.mse_path_[min_lcv[0]])

        mspe_lcv = mean_squared_error(
            model_lcv.predict(self.X_test), self.y_test)

        return loss_lcv, mspe_lcv, elapsed

    def run_sparseho(self, grad_solver='gd', algo_pick='imp_forw', n_steps=10, init_point=None, verbose=False):

        init_point_scale = self.scale_domain(init_point)

        estimator = Lasso(fit_intercept=False, max_iter=100, warm_start=True)
        model = WeightedLasso(estimator=estimator)
        sub_criterion = HeldOutMSE(None, None)
        criterion = CrossVal(sub_criterion, cv=self.kf)
        if algo_pick == 'imp_forw':
            algo = ImplicitForward()
        elif algo_pick == 'imp':
            algo = Implicit()
        else:
            raise ValueError("Undefined algo_pick.")

        monitor = Monitor()
        if grad_solver == 'gd':
            optimizer = GradientDescent(n_outer=n_steps, tol=self.tol_level,
                                        verbose=verbose, p_grad_norm=1.9)
        elif grad_solver == 'adam':
            optimizer = Adam(n_outer=n_steps, lr=0.11, verbose=verbose, tol=self.tol_level)
        elif grad_solver == 'line':
            optimizer = LineSearch(n_outer=n_steps, verbose=verbose, tol=self.tol_level)

        if init_point is None:
            grad_search(
                algo, criterion, model, optimizer, self.X_train, self.y_train,
                self.alpha_max/10*np.ones((self.X_train.shape[1],)), monitor)
        else:
            grad_search(
                algo, criterion, model, optimizer, self.X_train, self.y_train,
                init_point_scale, monitor)

        mspe = np.empty((n_steps,))
        config_all = np.empty((n_steps, self.n_features))
        reg_coef = np.empty((n_steps, self.n_features))

        for i in range(n_steps):
            estimator.weights = monitor.alphas[i]
            config_all[i, :] = monitor.alphas[i]
            estimator.fit(self.X_train, self.y_train)
            mspe[i] = mean_squared_error(estimator.predict(self.X_test), self.y_test)
            reg_coef[i, :] = estimator.coef_

        return monitor.objs, mspe/self.mspe_oracle, config_all, reg_coef, monitor.times
