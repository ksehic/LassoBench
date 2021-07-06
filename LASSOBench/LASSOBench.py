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

import timeit


class Synt_bench():
    def __init__(self, pick_bench=None, mf_opt=None, n_features=1280, n_samples=640,
                 snr_level=1, corr_level=0.6, n_nonzeros=10, tol_level=1e-4,
                 w_true=None, n_splits=5, test_size=0.15, seed=42):
        """
        Synthetic Benchmark that is used to test a HPO algorithm
        on different conditions. It is based on the cross-validation critetion.
        Args:
            pick_bench: select a predefined benchmark
            mf_opt: select the multi-fidelity framework
            n_features: the size of search space d>0 (i.e., the number of features)
            n_samples: the number of samples in a dataset
            snr_level: the level of noise with SNR=1 being very noisy and SNR=10 is almost noiseless.
            corr_level: the level of correlation withint features
            n_nonzeros: the number of nonzero elements in true reg coef betas
                        that can reduce or increase the sparsity of the solution
            w_true: the predefined reg coef betas numpy array with elemenets equal to n_features
            n_splits: the number of data splits for cross-validation
            test_size: the percentage of test data
            seed: the seed number
        Return:
            .evaluate:
                Arg: input_config - numpy array sampled within [-1, 1] with d number of elements
                Return: val_loss (the cross-validation loss for evaluate)
            .test: mspe_div (the mean-squared prediction error divided by the oracle error)
                  fscore (the F-measure for support recovery)
                  reg_coef (regression coefficients for input_config)
            .fidelity_evaluate: val_loss for each fidelity (use predefined fidelitis or use tol level to generate a fidelity)
            .run_hesbo: loss, mspe, fscore and elapsed for HesBO (high-dimensional BO algorithm)
            .run_LASSOCV: loss, mspe and elapsed for LassoCV
            .run_sparseho: loss, mspe, configuration steps, reg_coef and elapsed for Sparse-HO
        """

        if pick_bench is not None:
            if pick_bench == 'synt_low_eff_bench':
                n_features = 256
                n_samples = 128
                snr_level = 10
                corr_level = 0.6
                n_nonzeros = 8
            elif pick_bench == 'synt_high_eff_bench':
                n_features = 256
                n_samples = 128
                snr_level = 10
                corr_level = 0.6
                n_nonzeros = 20
            elif pick_bench == 'synt_high_noise_bench':
                n_features = 256
                n_samples = 128
                snr_level = 3
                corr_level = 0.6
                n_nonzeros = 8
            elif pick_bench == 'synt_high_corr_bench':
                n_features = 256
                n_samples = 128
                snr_level = 10
                corr_level = 0.9
                n_nonzeros = 8
            elif pick_bench == 'synt_hard_bench':
                n_features = 1280
                n_samples = 640
                snr_level = 1
                corr_level = 0.6
                n_nonzeros = 10
            else:
                raise ValueError(
                    "Please select one of the predefined benchmarks or creat your own.")

        if mf_opt is not None:
            if mf_opt == 'multi_continuous_bench':
                self.mf = 0
            elif mf_opt == 'multi_source_bench':
                self.mf = 1
            else:
                raise ValueError(
                    "Please select one of two mf options multi_continuous_bench or multi_source_bench.")

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

        self.eps_support = 1e-6

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

        estimator = Lasso(fit_intercept=False, max_iter=100, warm_start=False)
        model = WeightedLasso(estimator=estimator)
        monitor = Monitor()
        sub_criterion = HeldOutMSE(None, None)
        criterion = CrossVal(sub_criterion, cv=self.kf)
        val_loss = criterion.get_val(model, self.X_train, self.y_train,
                                     log_alpha=scaled_x,
                                     monitor=monitor, tol=self.tol_level)

        return val_loss

    def fidelity_evaluate(self, input_config, index_fidelity=None):

        if self.mf == 1:
            tol_range = np.geomspace(self.tol_level, 0.2, num=5)
            tol_budget = tol_range[index_fidelity]
        else:
            min_tol = -np.log(0.2)
            max_tol = -np.log(self.tol_level)
            tol_res = min_tol + index_fidelity*(max_tol - min_tol)
            tol_budget = np.exp(-tol_res)

        scaled_x = self.scale_domain(input_config)

        estimator = Lasso(fit_intercept=False, max_iter=100, warm_start=False)
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
    def __init__(self, pick_data=None, mf_opt=None, tol_level=1e-4, n_splits=5, test_size=0.15, seed=42):
        """
        Real world Benchmark that is used to test a HPO algorithm
        for datasets found in practice. It is based on the cross-validation critetion.
        Args:
            input_config: numpy array sampled within [-1, 1] with d number of elements
            pick_data: select real world dataset
            n_splits: the number of data splits for cross-validation
            seed: the seed number
        Return:
            .evaluate: val_loss (the cross-validation loss for evaluate)
            .test: mspe (the mean-squared prediction error)
                 reg_coef (regression coefficients for input_config)
            .fidelity_evaluate: val_loss for each fidelity (use predefined fidelitis or use tol level to generate a fidelity)
            .run_hesbo: loss, mspe and elapsed for HesBO (high-dimensional BO algorithm)
            .run_LASSOCV: loss, mspe and elapsed for LassoCV
            .run_sparseho: loss, mspe, configuration steps, reg_coef and elapsed for Sparse-HO
        """

        self.tol_level = tol_level

        if pick_data == 'diabetes':
            X, y = fetch_libsvm('diabetes_scale')
            alpha_scale = 1e5
        elif pick_data == 'breast-cancer':
            X, y = fetch_libsvm('breast-cancer_scale')
            alpha_scale = 1e5
        elif pick_data == 'leukemia':
            X, y = fetch_libsvm(pick_data)
            alpha_scale = 1e5
        elif pick_data == 'rcv1':
            X, y = fetch_libsvm('rcv1.binary')
            alpha_scale = 1e3
        elif pick_data == 'news20':
            X, y = fetch_libsvm('news20.binary')
            alpha_scale = 1e5
        else:
            raise ValueError("Unsupported dataset %s" % pick_data)

        if mf_opt is not None:
            if mf_opt == 'multi_continuous_bench':
                self.mf = 0
            elif mf_opt == 'multi_source_bench':
                self.mf = 1
            else:
                raise ValueError(
                    "Please select one of two mf options multi_continuous_bench or multi_source_bench.")

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

    def scale_domain(self, x):
        # Scaling the domain
        x_copy = np.copy(x)
        x_copy = x_copy * (
                self.log_alpha_max - self.log_alpha_min) / 2 + (
                    self.log_alpha_max + self.log_alpha_min) / 2
        return x_copy

    def evaluate(self, input_config):
        scaled_x = self.scale_domain(input_config)

        estimator = Lasso(fit_intercept=False, max_iter=100, warm_start=False)
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

        if self.mf == 1:
            tol_range = np.geomspace(self.tol_level, 0.2, num=5)
            tol_budget = tol_range[index_fidelity]
        else:
            min_tol = -np.log(0.2)
            max_tol = -np.log(self.tol_level)
            tol_res = min_tol + index_fidelity*(max_tol - min_tol)
            tol_budget = np.exp(-tol_res)

        scaled_x = self.scale_domain(input_config)

        estimator = Lasso(fit_intercept=False, max_iter=100, warm_start=False)
        model = WeightedLasso(estimator=estimator)
        monitor = Monitor()
        sub_criterion = HeldOutMSE(None, None)
        criterion = CrossVal(sub_criterion, cv=self.kf)
        val_loss = criterion.get_val(model, self.X_train, self.y_train,
                                     log_alpha=scaled_x,
                                     monitor=monitor, tol=tol_budget)

        return val_loss

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

        return monitor.objs, mspe, config_all, reg_coef, monitor.times
