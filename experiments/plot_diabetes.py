#!/usr/bin/python
"""
==================================================
LassoBench
High-Dimensional Hyperparameter
Optimization Benchmark
Contact: kenan.sehic@cs.lth.se

Example: Plotting the results for diabetes bench
=================================================
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


def plot_multi(data_mean, data_std, data_x, label_x, label_xaxis, xlimit, marker, mark_num, color_pick, num_index, ref_val=None, col_ref=None):
    """Plotting function for simple sythentic benchmark

    Args:
        data_mean (list): average results of each method
        data_std (list): standard deviation of each method
        data_x (list): number of iterations or wall-clock time
        label_x (list): labels for x axis for bar plot
        label_xaxis (str): label for x axis for simple regret
        xlimit (int): x axis limit
        marker (list): types of markes
        mark_num (list): marker's step
        color_pick (list): color
        num_index (list): order of methods
        ref_val (list, optional): Lasso-based baselines results. Defaults to None.
        col_ref (list, optional): Lasso-based baselines colors. Defaults to None.

    Returns:
        fig: returns the plot figure
    """

    plt.close('all')
    fig = plt.figure(figsize=(22, 10), constrained_layout=True)
    spec2 = gridspec.GridSpec(nrows=2, ncols=1, figure=fig, height_ratios=[2, 1])
    label_title = ['Diabetes benchmark ($d=8$ with $d_e=5$)']
    label_y = ['Best MSE', 'Final MSE']
    plot_l = []

    def min_acc(A):
        r = np.empty(len(A))
        r_ind = []
        for i in range(len(A)):
            if i > 0:
                if A[i] < r[i-1]:
                    r[i] = A[i]
                    j = i
                    r_ind.append(j)
                else:
                    r[i] = r[j]
                    r_ind.append(j)
            else:
                r[i] = A[i]
                j = i
                r_ind.append(j)
        return r

    for j in range(1):
        f_ax1 = fig.add_subplot(spec2[0, j])
        for i in range(len(data_mean)):
            l1, = f_ax1.plot(data_x[i][j], min_acc(data_mean[i][j]), '-',
                             color=color_pick[i], linewidth=2, marker=marker[i], markevery=mark_num[i],
                             markersize=15)
            plot_l.append(l1)

        if ref_val is not None:
            for i in range(len(ref_val)):
                l2 = f_ax1.axhline(y=ref_val[i][j], color=col_ref[i], linestyle='--', linewidth=4)
                plot_l.append(l2)

        plt.rcParams['font.family'] = 'Serif'
        # plt.yscale('log')
        plt.title(label_title[j], fontsize=28)
        plt.xlabel(label_xaxis + '\n''', fontsize=28)
        plt.xlim(0, xlimit)
        # plt.ylim(0, 1.2)
        if j == 0:
            plt.ylabel(label_y[0], fontsize=28)
        else:
            plt.ylabel(" ", fontsize=28)
        plt.rc('xtick', labelsize=28)
        plt.rc('ytick', labelsize=28)
        plt.grid()

        f_ax1 = fig.add_subplot(spec2[1, j])
        for i in range(len(data_mean)):
            min_index = np.argmin(data_mean[i][j])
            f_ax1.bar(np.array([num_index[i]]), np.min(data_mean[i][j]),
                yerr=data_std[i][j][min_index],
                align='center',
                alpha=1,
                color=color_pick[i],
                ecolor='black',
                capsize=10)
            print(np.min(data_mean[i][j]))
            print(data_std[i][j][min_index])
        if ref_val is not None:
            for i in range(len(ref_val)):
                f_ax1.bar(np.array([num_index[7 + i]]), ref_val[i][j],
                    yerr=0,
                    align='center',
                    alpha=1,
                    color=col_ref[i],
                    ecolor='black',
                    capsize=10)
                print(ref_val[i][j])

        plt.rcParams['font.family'] = 'Serif'
        plt.xticks(np.array(num_index), label_x, rotation='vertical', fontsize=28)
        if j == 0:
            plt.ylabel(label_y[1], fontsize=28)
        else:
            plt.ylabel(" ", fontsize=28)
        plt.title('  ', y=-0.5,pad=-14, fontsize=60)
        # plt.yscale('log')
        plt.ylim(0.6, 0.7)
        plt.grid()

    return fig, plot_l


# load results
with open('res_diabetes.pkl', 'rb') as f:
    res_cv, res_acv, mean_methods, std_methods, num_eval, _ = pickle.load(f)

# plot features
label_x = ['Random Search', 'Multi-start Sparse-HO', r'ALEBO $d_{\rm low}=3$', r'HeSBO $d_{\rm low}=2$', 'Hyperband', 'TuRBO', 'CMA-ES', 'LassoCV', 'AdaptiveLassoCV']
label_x2 = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
num_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
color_pick = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 'red', 'gold', 'cyan', 'tab:gray']
market_sym = ['P', 'X', 's', '^', 'D', '>', '8', '<', 'p', 'v']

ref_val = [res_cv, res_acv]
col_ref = ['k', 'b']

marker_step = [20, 20, 20, 20, 20, 20, 2, 2]
label_xaxis = 'Evaluations'
xlimit = 400

fig, l1 = plot_multi(mean_methods, std_methods, num_eval, label_x2, label_xaxis, xlimit, market_sym, marker_step, color_pick, num_index, ref_val, col_ref)
fig.legend(handles = l1 , labels=label_x, bbox_to_anchor=(0, -0.25, 1, 0.4), loc='upper center', mode='expand', numpoints=1, ncol=5, fancybox = True,
           fontsize=28, markerscale=2)
plt.savefig("diabetes_iter.pdf")
