#!/usr/bin/python
"""
==================================================
LassoBench
High-Dimensional Hyperparameter
Optimization Benchmark
Contact: kenan.sehic@cs.lth.se

Example: Plotting the results for real-world benchs (leukemia and rcv)
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
        plot_l0: list of methods
    """

    plt.close('all')
    fig = plt.figure(figsize=(28, 10), constrained_layout=True)
    spec2 = gridspec.GridSpec(nrows=2, ncols=2, figure=fig, height_ratios=[2, 1])
    label_title = [r'Leukemia benchmark ($d=7129$ with $\widehat{d}_e=22$)', r'RCV1 benchmark ($d=19959$ with $\widehat{d}_e=75$)']
    label_y = ['Best MSE (log)', 'Final MSE (log)']
    plot_l0 = []

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

    for j in range(2):
        plot_l = []
        print(j)
        print("next i")
        f_ax1 = fig.add_subplot(spec2[0, j])
        for i in range(len(data_mean[j])):
            print(i)
            l1, = f_ax1.plot(data_x[j][i][0], min_acc(data_mean[j][i][0]), '-',
                             color=color_pick[j][i], linewidth=2, marker=marker[j][i], markevery=mark_num[j][i],
                             markersize=15)
            plot_l.append(l1)

        if ref_val is not None:
            for i in range(len(ref_val[j])):
                l2 = f_ax1.axhline(y=ref_val[j][i], color=col_ref[j][i], linestyle='--', linewidth=4)
                plot_l.append(l2)

        plot_l0.append(plot_l)

        plt.rcParams['font.family'] = 'Serif'
        plt.title(label_title[j], fontsize=28)
        plt.xlabel(label_xaxis + '\n''', fontsize=28)
        plt.xlim(0, xlimit[j])
        if j == 0:
            plt.ylabel(label_y[0], fontsize=28)
        else:
            plt.ylabel("Best MSE", fontsize=28)

        if j == 0:
            plt.yscale('log')
        else:
            plt.ylim(0, 0.6)

        plt.rc('xtick', labelsize=28)
        plt.rc('ytick', labelsize=28)
        plt.grid()

        f_ax1 = fig.add_subplot(spec2[1, j])
        for i in range(len(data_mean[j])):
            min_index = np.argmin(data_mean[j][i][0])
            f_ax1.bar(np.array([num_index[j][i]]), np.min(data_mean[j][i][0]),
                yerr=data_std[j][i][0][min_index],
                align='center',
                alpha=1,
                color=color_pick[j][i],
                ecolor='black',
                capsize=10)
            print(np.min(data_mean[j][i][0]))
            print(data_std[j][i][0][min_index])
        if ref_val is not None:
            for i in range(len(ref_val[j])):
                f_ax1.bar(np.array([num_index[j][len(mark_num[j]) + i]]), ref_val[j][i],
                    yerr=0,
                    align='center',
                    alpha=1,
                    color=col_ref[j][i],
                    ecolor='black',
                    capsize=10)
                print(ref_val[j][i])

        if j == 0:
            plt.yscale('log')
            plt.ylim(10**-2, 10**0)
        else:
            plt.ylim(0, 0.4)

        plt.rcParams['font.family'] = 'Serif'
        plt.xticks(np.array(num_index[j]), label_x[j], rotation='vertical', fontsize=28)
        if j == 0:
            plt.ylabel(label_y[1], fontsize=28)
        else:
            plt.ylabel("Final MSE", fontsize=28)
        plt.title('  ', y=-0.5,pad=-14, fontsize=60)
        plt.grid()

    return fig, plot_l0


# load results for leukemia
with open('res_leu.pkl', 'rb') as f:
    res_cv_leu, res_acv_leu, mean_methods_leu, std_methods_leu, num_eval_leu, _ = pickle.load(f)

# plot features
label_x_leu = ['Random Search', 'Multi-start Sparse-HO', r'HeSBO $d_{\rm low}=2$', 'Hyperband', 'TuRBO', 'CMA-ES', 'LassoCV', 'AdaptiveLassoCV']
label_x2_leu = [' ', ' ', ' ', ' ', ' ',' ', ' ', ' ']
num_index_leu = [1, 2, 3, 4, 5, 6, 7, 8]
color_pick_leu = ['tab:orange', 'tab:green', 'tab:brown', 'tab:pink','red', 'gold']
market_sym_leu = ['P', 'X', '^', 'D', '>', '8', 'p']
marker_step_leu = [50, 50, 3, 3, 50, 3]

# LassoCV and AdaptiveLassoCV results
ref_val_leu = [res_cv_leu, res_acv_leu]
col_ref_leu = ['k', 'b']

## load results for rcv1
with open('res_rcv.pkl', 'rb') as f:
    res_cv_rcv, res_acv_rcv, mean_methods_rcv, std_methods_rcv, num_eval_rcv, _ = pickle.load(f)

# plot features
label_x_rcv = ['Random Search', 'Multi-start Sparse-HO', r'HeSBO $d_{\rm low}=2$', 'Hyperband', 'CMA-ES', 'LassoCV', 'AdaptiveLassoCV']
label_x2_rcv = [' ', ' ', ' ', ' ', ' ',' ', ' ']
num_index_rcv = [1, 2, 3, 4, 5, 6, 7]
color_pick_rcv = ['tab:orange', 'tab:green', 'tab:brown', 'tab:pink','gold']
market_sym_rcv = ['P', 'X', '^', 'D', '8', 'p', 'v']
marker_step_rcv = [50, 50, 50, 2, 2]

ref_val_rcv = [res_cv_rcv, res_acv_rcv]
col_ref_rcv = ['k', 'b']

# merge two results
mean_res = [mean_methods_leu, mean_methods_rcv]
std_res = [std_methods_leu, std_methods_rcv]
num_eval = [num_eval_leu, num_eval_rcv]
label_x0 = [label_x_leu, label_x_rcv]
label_x20 = [label_x2_leu, label_x2_rcv]
num_index0 = [num_index_leu, num_index_rcv]
color_pick0 = [color_pick_leu, color_pick_rcv]
market_sym0 = [market_sym_leu, market_sym_rcv]
marker_step0 = [marker_step_leu, marker_step_rcv]

# LassoCV and AdaptiveLassoCV results
ref_val = [ref_val_leu, ref_val_rcv]
col_ref = [col_ref_leu, col_ref_rcv]

xlimit0 = [2000, 1000]

label_xaxis = 'Evaluations'

fig, l1 = plot_multi(mean_res, std_res, num_eval, label_x20, label_xaxis, xlimit0, market_sym0, marker_step0, color_pick0, num_index0, ref_val, col_ref)
label_x_all = ['Random Search', 'Multi-start Sparse-HO', r'HeSBO $d_{\rm low}=2$', 'Hyperband', 'TuRBO', 'CMA-ES', 'LassoCV', 'AdaptiveLassoCV']
l_all = []

for i in range(len(label_x_rcv)):
    if i == 4:
        l_all.append(l1[0][i])
    l_all.append(l1[1][i])

fig.legend(handles = l_all , labels=label_x_all, bbox_to_anchor=(0.01,-0.25,0.995, 0.4), loc='upper center', mode='expand', numpoints=1, ncol=4, fancybox = True,
           fontsize=28, markerscale=2)

plt.savefig("plot_leu_rcv22222.pdf")