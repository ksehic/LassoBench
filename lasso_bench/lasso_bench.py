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


class SyntheticBenchmark:
    """
    Creating a synthetic benchmark for a HPO algorithm.

    ...

    Attributes
    ----------
    pick_bench : str
        name of a predefined bench such as
        synt_low_eff_bench, synt_high_eff_bench, synt_high_noise_bench,
        synt_high_corr_bench and synt_hard_bench
    mf_opt : str, optional
        name of a multi-fidelity framework
        multi_continuous_bench or multi_source_bench
    n_features : int, optional
        number of features in design matrix i.e. the dimension of search space
    n_samples : int, optional
        number of samples in design matrix
    snr_level : int, optional
        level of noise, 1 very noisy 10 noiseless
    corr_level : int, optional
        correlation between features in design matrix
    n_nonzeros: int, optional
        number of nonzero elements in true reg coef
    tol_level: int, optional
        tolerance level for inner opt part
    w_true: array, optional
        custimized true regression coefficients
    n_splits: int, optional
        number of splits in CV
    test_size: int, optional
        percentage of test data
    seed: int, optional
        seed number

    Methods
    -------
    evaluate(input_config):
        Return cross-validation loss divided by oracle for configuration.
    fidelity_evaluate(input_config, index_fidelity=None):
        Return cross-validation loss for configuration and fidelity index.
    test(input_config):
        Return post-processing metrics MSPE divided by oracle error,
        Fscore and reg coef for configuration.
    run_LASSOCV(n_alphas=100):
        Running baseline LASSOCV and return loss, mspe divided by oracle,
        fsore, best-found 1D config and time elapsed.
    run_sparseho(grad_solver='gd', algo_pick='imp_forw', n_steps=10, init_point=None, verbose=False):
        Running basedline Sparse-HO and return loss, mspe divided by oracle,
        fscore and time elapsed.
    """

    def __init__(self, pick_bench=None, mf_opt=None, n_features=1280, n_samples=640,
                 snr_level=1, corr_level=0.6, n_nonzeros=10, tol_level=1e-4,
                 w_true=None, n_splits=5, test_size=0.15, seed=42):
        """
        Constructs all the necessary attributes for synt bench.

        Parameters
        ----------
            pick_bench : str
                name of a predefined bench such as
                synt_low_eff_bench, synt_high_eff_bench, synt_high_noise_bench,
                synt_high_corr_bench and synt_hard_bench
            mf_opt : str, optional
                name of a multi-fidelity framework
                multi_continuous_bench or multi_source_bench
            n_features : int, optional
                number of features in design matrix i.e. the dimension of search space
            n_samples : int, optional
                number of samples in design matrix
            snr_level : int, optional
                level of noise, 1 very noisy 10 noiseless
            corr_level : int, optional
                correlation between features in design matrix
            n_nonzeros: int, optional
                number of nonzero elements in true reg coef
            tol_level: int, optional
                tolerance level for inner opt part
            w_true: array, optional
                custimized true regression coefficients
            n_splits: int, optional
                number of splits in CV
            test_size: int, optional
                percentage of test data
            seed: int, optional
                seed number
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

        self.mf = 2

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
        """
        Evaluate configuration for synt bench

        Parameters
        ----------
        input_config : array size of n_features

        Returns
        -------
        Cross-validation Loss divided by oracle. The goal is to be close or less than 1.
        """
        scaled_x = self.scale_domain(input_config)

        estimator = Lasso(fit_intercept=False, max_iter=100, warm_start=False)
        model = WeightedLasso(estimator=estimator)
        monitor = Monitor()
        sub_criterion = HeldOutMSE(None, None)
        criterion = CrossVal(sub_criterion, cv=self.kf)
        val_loss = criterion.get_val(model, self.X_train, self.y_train,
                                     log_alpha=scaled_x,
                                     monitor=monitor, tol=self.tol_level)

        return val_loss/self.mspe_oracle

    def fidelity_evaluate(self, input_config, index_fidelity=None):
        """
        Return cross-validation loss for selected fidelity.

        Parameters
        ----------
        input_config : array size of n_features
        index_fidelity : int, optional
            If mf_opt is selected, then selecting which fidelity to evaluate. (default is None)
            For multi_source_bench, index_fidelity is a dicreate par between 0 and 5.

        Returns
        -------
        Cross-validation Loss divided by oracle. The goal is to be close or less than 1.
        """
        if self.mf == 1:
            tol_range = np.geomspace(self.tol_level, 0.2, num=5)
            tol_budget = tol_range[index_fidelity]
        elif self.mf == 0:
            min_tol = -np.log(0.2)
            max_tol = -np.log(self.tol_level)
            tol_res = min_tol + index_fidelity*(max_tol - min_tol)
            tol_budget = np.exp(-tol_res)
        else:
            raise ValueError(
                "Please select one of two mf options multi_continuous_bench or multi_source_bench.")

        scaled_x = self.scale_domain(input_config)

        estimator = Lasso(fit_intercept=False, max_iter=100, warm_start=False)
        model = WeightedLasso(estimator=estimator)
        monitor = Monitor()
        sub_criterion = HeldOutMSE(None, None)
        criterion = CrossVal(sub_criterion, cv=self.kf)
        val_loss = criterion.get_val(model, self.X_train, self.y_train,
                                     log_alpha=scaled_x,
                                     monitor=monitor, tol=tol_budget)

        return val_loss/self.mspe_oracle

    def test(self, input_config):
        """
        Post-processing metrics MSPE and Fscore

        Parameters
        ----------
        input_config : array size of n_features

        Returns
        -------
        MSPE divided by oracle and Fscore
        """
        scaled_x = self.scale_domain(input_config)
        estimator = Lasso(fit_intercept=False, max_iter=100, warm_start=False)
        estimator.weights = np.exp(scaled_x)
        estimator.fit(self.X_train, self.y_train)
        reg_coef = estimator.coef_
        coef_hpo_support = np.abs(estimator.coef_) > self.eps_support
        fscore = f1_score(self.coef_true_support, coef_hpo_support)
        mspe = mean_squared_error(estimator.predict(self.X_test), self.y_test)
        mspe_div = mspe/self.mspe_oracle

        return mspe_div, fscore

    def run_LASSOCV(self, n_alphas=100):
        """
        Running baseline LASSOCV

        Parameters
        ----------
        n_alphas : int, optional
            The number of grid points in 1D optimization (default is 100)

        Returns
        -------
        Cross-validation Loss, MSPE divided by oracle, fscore and time elapsed
        """

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

        return loss_lcv, mspe_lcv/self.mspe_oracle, fscore, elapsed

    def run_sparseho(self, grad_solver='gd', algo_pick='imp_forw', n_steps=10, init_point=None):
        """
        Running baseline Sparse-HO

        Parameters
        ----------
        grad_solver : str, optional
            Selecting which gradient solver to use gradient descent 'gd', 'adam' or 'line' as line search (default is gd)
        algo_pick   : str, optional
            Selecting which diff solver to use imp_forw or imp (default is imp_forw)
        n_steps     : int, optional
            Number of optimization steps (default is 10)
        init_point  : array, optional
            First guess (default is None)

        Returns
        -------
        Cross-validation loss, MSPE divided by oracle, fscore and time elapsed
        """

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
                                        verbose=False, p_grad_norm=1.9)
        elif grad_solver == 'adam':
            optimizer = Adam(n_outer=n_steps, lr=0.11, verbose=False, tol=self.tol_level)
        elif grad_solver == 'line':
            optimizer = LineSearch(n_outer=n_steps, verbose=False, tol=self.tol_level)

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

        return monitor.objs, mspe/self.mspe_oracle, fscore, monitor.times



class RealworldBenchmark():
    """
    Creating a real-world benchmark for a HPO algorithm.

    ...

    Attributes
    ----------
    pick_data : str
        name of dataset such as
        diabetes, breast_cancer, leukemia, rcv1, news20
    mf_opt : str, optional
        name of a multi-fidelity framework
        multi_continuous_bench or multi_source_bench
    tol_level: int, optional
        tolerance level for inner opt part
    n_splits: int, optional
        number of splits in CV
    test_size: int, optional
        percentage of test data
    seed: int, optional
        seed number

    Methods
    -------
    evaluate(input_config):
        Return cross-validation loss for configuration.
    fidelity_evaluate(input_config, index_fidelity=None):
        Return cross-validation loss for configuration and fidelity index.
    test(input_config):
        Return post-processing metric MSPE.
    run_LASSOCV(n_alphas=100):
        Running baseline LASSOCV and return loss, MSPE and time elapsed.
    run_sparseho(grad_solver='gd', algo_pick='imp_forw', n_steps=10, init_point=None, verbose=False):
        Running basedline Sparse-HO and return loss, MSPE and time elapsed.
    """
    def __init__(self, pick_data=None, mf_opt=None, tol_level=1e-4, n_splits=5, test_size=0.15, seed=42):
        """
        Constructs all the necessary attributes for real-world bench.

        Parameters
        ----------
            pick_data : str
                name of dataset such as
                diabetes, breast_cancer, leukemia, rcv1, news20
            mf_opt : str, optional
                name of a multi-fidelity framework
                multi_continuous_bench or multi_source_bench
            tol_level: int, optional
                tolerance level for inner opt part
            n_splits: int, optional
                number of splits in CV
            test_size: int, optional
                percentage of test data
            seed: int, optional
                seed number
        """

        self.tol_level = tol_level

        if pick_data == 'diabetes':
            X, y = fetch_libsvm('diabetes_scale')
            alpha_scale = 1e5
        elif pick_data == 'breast_cancer':
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

        self.mf = 2

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
        """
        Evaluate configuration for synt bench

        Parameters
        ----------
        input_config : array size of n_features

        Returns
        -------
        Cross-validation Loss
        """
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
        """
        Return cross-validation loss for selected fidelity.

        Parameters
        ----------
        input_config : array size of n_features
        index_fidelity : int, optional
            If mf_opt is selected, then selecting which fidelity to evaluate. (default is None)
            For multi_source_bench, index_fidelity is a dicreate par between 0 and 5.

        Returns
        -------
        Cross-validation Loss
        """

        if self.mf == 1:
            tol_range = np.geomspace(self.tol_level, 0.2, num=5)
            tol_budget = tol_range[index_fidelity]
        elif self.mf == 0:
            min_tol = -np.log(0.2)
            max_tol = -np.log(self.tol_level)
            tol_res = min_tol + index_fidelity*(max_tol - min_tol)
            tol_budget = np.exp(-tol_res)
        else:
            raise ValueError(
                "Please select one of two mf options multi_continuous_bench or multi_source_bench.")

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
        """
        Post-processing metrics MSPE and Fscore

        Parameters
        ----------
        input_config : array size of n_features

        Returns
        -------
        MSPE
        """
        scaled_x = self.scale_domain(input_config)
        estimator = Lasso(fit_intercept=False, max_iter=100, warm_start=False)
        estimator.weights = np.exp(scaled_x)
        estimator.fit(self.X_train, self.y_train)
        mspe = mean_squared_error(estimator.predict(self.X_test), self.y_test)

        return mspe

    def run_LASSOCV(self, n_alphas=100):
        """
        Running baseline LASSOCV

        Parameters
        ----------
        n_alphas : int, optional
            The number of grid points in 1D optimization (default is 100)

        Returns
        -------
        Cross-validation Loss, MSPE divided by oracle and time elapsed
        """

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

    def run_sparseho(self, grad_solver='gd', algo_pick='imp_forw', n_steps=10, init_point=None):
        """
        Running baseline Sparse-HO

        Parameters
        ----------
        grad_solver : str, optional
            Selecting which gradient solver to use gradient descent 'gd', 'adam' or 'line' as line search (default is gd)
        algo_pick   : str, optional
            Selecting which diff solver to use imp_forw or imp (default is imp_forw)
        n_steps     : int, optional
            Number of optimization steps (default is 10)
        init_point  : array, optional
            First guess (default is None)

        Returns
        -------
        Cross-validation loss, MSPE divided by oracle and time elapsed
        """

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
                                        verbose=False, p_grad_norm=1.9)
        elif grad_solver == 'adam':
            optimizer = Adam(n_outer=n_steps, lr=0.11, verbose=False, tol=self.tol_level)
        elif grad_solver == 'line':
            optimizer = LineSearch(n_outer=n_steps, verbose=False, tol=self.tol_level)

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

        return monitor.objs, mspe, monitor.times
