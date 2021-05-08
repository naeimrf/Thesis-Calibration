import math

import matplotlib.pyplot as plt
import numpy as np, seaborn as sns, pandas as pd
import local_utility as lu
from local_setup import *
import os, statistics
from skopt.sampler import *
from scipy import stats
from skopt.space import Space
from SALib.sample import latin
from sklearn.manifold import TSNE
from matplotlib.ticker import NullFormatter
import matplotlib.cm as cmap
import time


def plot_side_by_side(theoretical_freq, theoretical_xlabels, lpc_freq, discrete, params, message, color):
    groups = len(params)
    x = np.arange(discrete)  # position of groups, using X to align the bars side by side

    width = 0.75
    fig, axs = plt.subplots(1, groups, figsize=(7, 4.5), sharey='all')

    for n, ax in enumerate(axs):
        ax.bar(x, theoretical_freq[n], color='gray', alpha=0.7,
               width=width, align='center', edgecolor='white')
        ax.bar(x, lpc_freq[n], color=color, alpha=0.5,
               width=width, align='center', edgecolor='white')

        ax.set_title(params[n], fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(theoretical_xlabels[n], rotation='vertical', fontsize=9)

    # Real numbers in simulation are based on Latin Hyper Cube method!
    fig.text(0.04, 0.5, f'Theoretical frequency of uncertain parameters ({round((1 / discrete), 3)})',
             va='center', rotation='vertical', fontsize=10)

    plt.ylim(0, np.max(np.array(lpc_freq)) * 1.1)  # set y range 10% more than maximum value of frequencies
    plt.yticks(size=7)
    fig.canvas.set_window_title(message)
    plt.show()


def plot_prior_distributions(params, ranges, buildings, discrete, per_building=True, SALib=True, **kwargs):
    """
    This method plots the actual prior samplings over the theoretical distribution per building
    """
    print(f'\t- The start of plot_prior_distributions method ...')
    groups = len(params)
    theoretical_freq = [[1 / discrete] * discrete] * groups
    lhc = kwargs.get('LHC')  # Latin Hyper Cube Frequencies

    lhc_xlabels_all_buildings = [[] for _ in range(groups)]

    for b_nbr in buildings:
        theoretical_xlabels = []
        for pr in ranges:
            if pr[0] < 0.01:
                theoretical_xlabels.append(list(np.round(np.linspace(pr[0], pr[1], discrete), 3)))
            else:
                theoretical_xlabels.append(list(np.round(np.linspace(pr[0], pr[1], discrete), 2)))

        lhc_xlabels = []

        for i, param in enumerate(params):
            lhc_xlabels.append(lhc[b_nbr][param])
            lhc_xlabels_all_buildings[i] += lhc[b_nbr][param]

    if per_building:
        for b_nbr in buildings:
            if not SALib:
                message = f'Constant probabilistic parameters vs SALib-LHC in building {b_nbr}'
            else:
                message = f'Constant probabilistic parameters vs Centered-LHC in building {b_nbr}'
            lhc_freq = lu.get_projected_freq(theoretical_xlabels, lhc_xlabels, groups)
            plot_side_by_side(theoretical_freq, theoretical_xlabels, lhc_freq, discrete, params, message, color='blue')
    else:
        if not SALib:
            message = f'Constant probabilistic parameters for all buildings with SALib-LHC'
        else:
            message = f'Constant probabilistic parameters for all buildings with Centered-LHC'
        lhc_freq = lu.get_projected_freq(theoretical_xlabels, lhc_xlabels_all_buildings, groups)
        plot_side_by_side(theoretical_freq, theoretical_xlabels, lhc_freq, discrete, params, message, color='blue')

    print(f'\t+ plot_prior_distributions method is over!')


def plot_calibrated_parameters(data_frame, ranges, discrete, samples_nbr, alpha, plot=True):
    if 'Buildings' in data_frame.columns:
        del data_frame['Buildings']

    params = data_frame.columns.values.tolist()
    groups = len(params)

    theoretical_xlabels = []
    for pr in ranges:
        if pr[0] < 0.01:
            theoretical_xlabels.append(list(np.round(np.linspace(pr[0], pr[1], discrete), 3)))
        else:
            theoretical_xlabels.append(list(np.round(np.linspace(pr[0], pr[1], discrete), 2)))

    lhc_xlabels = []
    for param in params:
        tmp = data_frame[param].tolist()
        lhc_xlabels.append(tmp)

    theoretical_freq = [[1 / discrete] * discrete] * groups
    message = f'Prior and posterior (calibrated) distributions, simulations:{samples_nbr}, alpha:{alpha}'
    lhc_freq = lu.get_projected_freq(theoretical_xlabels, lhc_xlabels, groups)
    if plot:
        plot_side_by_side(theoretical_freq, theoretical_xlabels, lhc_freq, discrete, params, message, color='green')

    return lhc_freq


def passed_cases_one_building(error_dict, nbr, alpha):
    # plot error values
    error_to_plot = np.fromiter(error_dict[nbr].values(), dtype=float)
    passed_size = len(error_to_plot[error_to_plot < alpha])
    sim_size = range(len(error_to_plot))

    plt.figure(figsize=(7, 4), dpi=80)
    error_colors = np.where(error_to_plot < alpha, 'g', 'k')
    plt.scatter(sim_size, error_to_plot, c=error_colors)
    plt.axhline(y=alpha, color='r', linestyle='--')

    y = min(error_dict[nbr].values())
    plt.annotate(str(round(y, 2)) + "%", (np.argmin(error_to_plot), y),
                 xytext=(0, 10), textcoords="offset points", ha='center')

    plt.xlabel(f'Number of simulations for building {nbr}')
    plt.ylabel('Percentage of Error')
    fig = plt.gcf()
    perc = round((passed_size / len(error_to_plot)) * 100, 2)
    fig.canvas.set_window_title(f'{passed_size}/{len(error_to_plot)} ({perc}%) '
                                f'of tests with error lower than {alpha}%')
    plt.tight_layout()
    plt.show()


def passed_cases_all_buildings(error_dict, alpha):
    building_ids = error_dict.keys()

    all_errors_list = []
    min_error_per_building = {}
    for id in building_ids:
        tmp = list(np.fromiter(error_dict[id].values(), dtype=float))
        all_errors_list += tmp
        min_value = min(error_dict[id].values())
        min_idx = tmp.index(min(tmp)) + len(error_dict[id]) * list(building_ids).index(id)
        min_error_per_building.update({id: (min_value, min_idx)})
    all_errors_list = np.array(all_errors_list)

    passed_size = len(all_errors_list[all_errors_list < alpha])
    sim_size = range(len(all_errors_list))

    plt.figure(figsize=(7, 4), dpi=80)
    error_colors = np.where(all_errors_list < alpha, 'g', 'k')
    plt.scatter(sim_size, all_errors_list, c=error_colors)
    plt.axhline(y=alpha, color='r', linestyle='--')

    for id in building_ids:
        min_value = min_error_per_building[id][0]
        min_idx = min_error_per_building[id][1]
        color = 'k' if min_value <= alpha else 'r'
        plt.annotate((str(round(min_value, 2)) + "%", id), (min_idx, min_value),
                     xytext=(0, 10), textcoords="offset points", ha='center', color=color)

    plt.xlabel(f'Number of all simulations for buildings: {[i for i in building_ids]}')
    plt.ylabel('Percentage of Error')
    fig = plt.gcf()
    perc = round((passed_size / len(all_errors_list)) * 100, 2)
    fig.canvas.set_window_title(f'{passed_size}/{len(all_errors_list)} ({perc}%) '
                                f'of tests with error lower than {alpha}%')
    plt.tight_layout()
    plt.show()


def plot_metered_vs_simulated_energies(metered, simulated, bins=20):
    metered_mean = statistics.mean(metered)
    simulated_mean = statistics.mean(simulated)

    # From scipy documentation: if the K-S statistic is small or the p-value is high, then we cannot reject
    # the hypothesis that the distributions of the two samples are the same.
    # If p-value is lower than a=0.05 or 0.01, then it is very probable that the two distributions are different.
    ks_test = stats.ks_2samp(metered, simulated)

    plt.figure(figsize=(6, 4))
    plt.hist(metered, bins=bins, label='Metered', alpha=0.5, color='red', histtype='step')
    plt.hist(simulated, bins=bins, label='Simulated', alpha=0.8, color='gray', histtype='barstacked')
    plt.axvline(x=metered_mean, ls='--', alpha=0.5, color='red', label='Metered mean')
    plt.axvline(x=simulated_mean, ls='-', alpha=0.5, color='black', label='Simulated mean')
    plt.title(
        f'Metered mean:{round(metered_mean, 2)} vs Simulated_mean:{round(simulated_mean, 2)}\nKS-p_value:{round(ks_test[1], 2)}',
        color="k", fontsize=9)
    plt.xlabel('EUI(kWh/m2)')
    plt.ylabel('Frequency')
    plt.legend(loc='best')  # bbox_to_anchor=(1.04, 1)
    # plt.xticks(np.arange(min(simulated), max(simulated)+1, 5.0))

    plt.tight_layout()
    plt.show()


def plot_combinations(var_names, bounds, combs, points, sample_nbr):
    info = dict(zip(var_names, bounds))
    var_nbr = len(var_names)
    tmp = []
    for var in var_names:
        for i in range(sample_nbr):
            tmp.append(var)
    var_names = tmp

    freq = {}
    for point in points:
        for p in point:
            freq[p] = sum(x.count(p) for x in combs)

    keys = list(freq.keys())
    keys = [round(elem, 2) for elem in keys]
    vals = list(freq.values())
    table = {'Variables': var_names, 'Range of middle points': keys, 'Frequency': vals}
    table = pd.DataFrame.from_dict(table)

    plt.figure(figsize=(8, 4))
    g = sns.barplot(x='Range of middle points', y='Frequency', data=table, hue='Variables', dodge=False)
    g.set_xticklabels(g.get_xticklabels(), rotation=90)

    fig = plt.gcf()
    fig.canvas.set_window_title("Frequency of real samples with skopt-Centered-LHC ")

    to_print = ''
    for i in range(var_nbr - 1):
        to_print += str(sample_nbr) + 'x'
    to_print += str(sample_nbr)
    plt.title(f'{var_nbr} variables & {sample_nbr} samples: {to_print} '
              f'= {sample_nbr ** var_nbr} combinations!\n{info}', fontsize=9)
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2)
    plt.tight_layout()
    plt.show()


def plot_joint_distributions(data_frame):
    del data_frame['Buildings']

    sns.set_theme(style="white")
    # sns.palplot(sns.color_palette("BrBG", 12))
    cmap = sns.cubehelix_palette(light=1.5, as_cmap=True)

    header = list(data_frame.columns)
    graph = {}
    for i in range(len(header) - 1):
        graph[f'g{i + 1}'] = None

    i = 0
    for key in graph:
        graph[key] = sns.JointGrid(data=data_frame, x=header[0], y=header[i + 1], space=0)
        graph[key].set_axis_labels(fontsize=12)
        graph[key].plot_joint(sns.kdeplot, fill=True, clip=((Bounds[0]), (Bounds[i + 1])),
                              thresh=0, levels=5, cmap=cmap).plot_joint(sns.scatterplot)
        graph[key].plot_marginals(sns.histplot, color="gray", alpha=1, bins=5)
        graph[key].savefig(f'{key}.png')
        plt.close(fig=None)
        i += 1

    fig, ax = plt.subplots(nrows=1, ncols=len(header) - 1, figsize=(10, 4))
    i = 0
    for key in graph.keys():
        ax[i].imshow(plt.imread(f'{key}.png'))
        i += 1

    fig.canvas.set_window_title(f'A kernel density estimate of validated combination of parameters')
    [ax.set_axis_off() for ax in ax.ravel()]
    plt.tight_layout()
    plt.show()

    for key in graph.keys():
        os.remove(f'{key}.png')


# METHODS RELATED TO RANDOM SAMPLING COMPARISON  - - - - - - - -
def manual_lhc(n):
    # https://www.youtube.com/watch?v=r6rp-Qxc9xI
    lower_band = np.arange(0, n) / n
    upper_band = np.arange(1, n + 1) / n
    points = np.random.uniform(low=lower_band, high=upper_band, size=[2, n]).T
    np.random.shuffle(points[:, 1])

    plt.figure(figsize=[5, 5])
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.scatter(points[:, 0], points[:, 1], c='r')

    for i in np.arange(0, 1, 1 / n):
        plt.axvline(i, linestyle=':', color='gray')
        plt.axhline(i, linestyle=':', color='gray')
    plt.show()


def rand_plot_1d(xx, nbr_samples, message, space, color='b*'):
    plt.plot(xx, np.zeros_like(xx), color)
    plt.title(message, fontsize=10)
    plt.xticks(size=8)
    plt.yticks(size=6)
    for i in np.arange(space.bounds[0][0], space.bounds[0][1],
                       (space.bounds[0][1] - space.bounds[0][0]) / nbr_samples):
        plt.axvline(i, linestyle=':', color='gray')


def generate_randomness(dimensions, nbr_samples, space):
    """
    The purpose of this method is to compare randomness of different algorithms in sikit-optimization package
    with Latin HyperCube msampler in SALib
    https://scikit-optimize.github.io/stable/auto_examples/sampler/initial-sampling-method.html#pdist-boxplot-of-all-methods
    https://github.com/SALib/SALib
    """
    fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(9, 4), sharex='all', sharey='all')

    # row, column, nbr
    plt.subplot(5, 2, 1)
    sobol = Sobol()
    sb = sobol.generate(dimensions, nbr_samples)
    rand_plot_1d(sb, nbr_samples, 'skopt-Sobol', space)

    plt.subplot(5, 2, 2)
    lhs = Lhs(lhs_type="classic", criterion=None)
    classic = lhs.generate(space.dimensions, nbr_samples)
    rand_plot_1d(classic, nbr_samples, 'skopt-Classic LHC', space)

    plt.subplot(5, 2, 3)
    lhs = Lhs(lhs_type="centered", criterion=None)
    centered = lhs.generate(space.dimensions, nbr_samples)
    rand_plot_1d(centered, nbr_samples, 'skopt-Centered LHC', space, color='gh')

    plt.subplot(5, 2, 4)
    lhs = Lhs(criterion="maximin", iterations=1000)
    maximin = lhs.generate(space.dimensions, nbr_samples)
    rand_plot_1d(maximin, nbr_samples, 'skopt-Maximin LHC', space)

    plt.subplot(5, 2, 5)
    lhs = Lhs(criterion="correlation", iterations=1000)
    correlation = lhs.generate(space.dimensions, nbr_samples)
    rand_plot_1d(correlation, nbr_samples, 'skopt-Correlation LHC', space)

    plt.subplot(5, 2, 6)
    lhs = Lhs(criterion="correlation", iterations=1000)
    ratio = lhs.generate(space.dimensions, nbr_samples)
    rand_plot_1d(ratio, nbr_samples, 'skopt-Ratio LHC', space)

    plt.subplot(5, 2, 7)
    halton = Halton()
    hal = halton.generate(space.dimensions, nbr_samples)
    rand_plot_1d(hal, nbr_samples, 'skopt-Halton', space)

    plt.subplot(5, 2, 8)
    hammersly = Hammersly()
    ham = hammersly.generate(space.dimensions, nbr_samples)
    rand_plot_1d(ham, nbr_samples, 'skopt-Hammersly', space)

    plt.subplot(5, 2, 9)
    grid = Grid(border="exclude", use_full_layout=False)
    gr = grid.generate(space.dimensions, nbr_samples)
    rand_plot_1d(gr, nbr_samples, 'skopt-Grid', space)

    plt.subplot(5, 2, 10)
    problem = {'names': 'x1', 'bounds': [[space.bounds[0][0], space.bounds[0][1]]], 'num_vars': 1}
    salib = latin.sample(problem, nbr_samples)
    rand_plot_1d(salib, nbr_samples, 'SALib-LHC', space, color='mx')

    fig.canvas.set_window_title('Comparison of Quasi-Random Number Generators')
    plt.tight_layout()
    plt.show()

    return centered, salib


def compare2_methods(centered, salib, dimensions, sample_nbr, name1, name2):
    data = [centered, salib]
    names = [name1, name2]
    makers = ['b*', 'r+']
    distances = [15, -15]

    flat_data = []
    for d in data:
        tmp = []
        for item in d:
            tmp.extend(item)
        flat_data.append(sorted(tmp))

    for d, n, m, dis in zip(flat_data, names, makers, distances):
        plt.plot(range(len(d)), d, m, label=n)

        for x, y in zip(range(len(d)), d):
            label = str(round(y, 2))
            plt.annotate(label,  # this is the text
                         (x, y),  # this is the point to label
                         textcoords="offset points",  # how to position the text
                         xytext=(dis, 0),  # distance from text to points (x,y)
                         ha='center',
                         fontsize=8)

    nbrs = range(len(flat_data[0]))
    scope = ((dimensions[0].bounds[1] - dimensions[0].bounds[0]) / sample_nbr) / 2
    plt.errorbar(nbrs, flat_data[0], yerr=scope, ecolor='lightsteelblue', ls='none', capsize=4)
    plt.xticks(nbrs)
    plt.xlabel('Number of samples')
    plt.ylabel('Random value by the method')

    # plt.grid()
    plt.legend()
    fig = plt.gcf()
    fig.canvas.set_window_title('Illustration of samples for two Quasi-Random Number Generators')
    plt.show()


def plot_recursive_improvement(result_dict):
    iterations = result_dict.keys()
    order_dict = {}
    for b in BuildNum:
        tmp = []
        for value in result_dict.values():
            tmp.append(value[b])
            order_dict.update({b: tmp})
    order_dict, len(order_dict)

    colors = list("rgbcmyk")
    x = iterations
    for y in order_dict.values():
        plt.plot(x, y, color=colors.pop(), ls="--", marker="x")

    plt.suptitle('Building error reduction with recursive calibration')
    plt.xlabel('Number of iterations')
    plt.ylabel('Error')
    plt.legend(order_dict.keys())
    plt.show()


def make_t_SNE_plot(calib_params):
    # The method is inspired by Narine Kokhlikyan's code in scikit-learn website:
    # https://scikit-learn.org/stable/auto_examples/manifold/plot_t_sne_perplexity.html#sphx-glr-auto-examples-manifold-plot-t-sne-perplexity-py
    print(f"\t- The start of t_SNE_plot method ...")
    n_samples = len(calib_params.index)
    target = calib_params["Buildings"].to_numpy()
    df_tmp = calib_params.drop('Buildings', 1)  # 0 for rows and 1 for columns
    parameters = len(df_tmp.columns)

    if n_samples <= 100:
        method = 'exact'
    else:
        method = 'barnes_hut'
    print(
        f"\t-> Calibration in buildings: {set(target)} consists of {n_samples} combinations of {parameters} parameters")
    perplexities = (5, 30, 50, 100)
    learning_rates = (100, 200, 400)
    # row: learning_rates, column: perplexities, figure-size
    (fig, subplots) = plt.subplots(len(learning_rates), len(perplexities),
                                   sharex='all', sharey='all', figsize=(9, 5))
    t00 = time.time()
    for i, perplexity in enumerate(perplexities):
        for j, learning_rate in enumerate(learning_rates):
            ax = subplots[j][i]
            t0 = time.time()
            # fit and transform with TSNE
            t_sne = TSNE(n_components=2, perplexity=perplexity, method=method,
                         learning_rate=learning_rate, n_iter=5000, verbose=0)
            # project the data in 2D
            y = t_sne.fit_transform(df_tmp)
            if i == 0:
                ax.set_ylabel(learning_rate, rotation=90, size='small')
            if j == len(learning_rates) - 1:
                ax.set_xlabel(perplexity, rotation=0, size='small')

            for t in target:
                ax.scatter(y[target == t, 0], y[target == t, 1], cmap=cmap, s=5)
                ax.xaxis.set_major_formatter(NullFormatter())
                ax.yaxis.set_major_formatter(NullFormatter())

            t1 = time.time()
            print(f"\t-> t_SNE plot, perplexity:{perplexity} and learning_rate:"
                  f"{learning_rate} in {round((t1 - t0), 2)}s")

    t11 = time.time()
    fig.canvas.set_window_title('t-SNE plots for calibrated parameters, columns: perplexities, rows: learning_rates')
    plt.tight_layout()
    plt.show()
    print(f"\t+ The t_SNE_plot method is over in {round((t11 - t00), 2)}s!")


def distribution_plots_calibrated(calib_params, Bounds, discrete, NbRuns):
    calib_params = calib_params.drop('Buildings', 1)

    nbr_params = len(list(calib_params.columns))
    theoretical_freq = [[1 / discrete] * discrete] * nbr_params

    f, axes = plt.subplots(1, nbr_params, figsize=(10, 3), sharey='all')
    sns.set(style="white", palette="muted", color_codes=True)

    for param, x in zip(calib_params.columns, range(nbr_params)):
        axes[x].axhline(y=theoretical_freq[x][0], linewidth=1.5,
                        linestyle="--", color='gray') # xmin=Bounds[x][0], xmax=Bounds[x][1]
        sns.histplot(calib_params[param], kde=True, ax=axes[x], stat='probability',
                     line_kws={"lw": 1.5}, color="darkgreen", bins=discrete)
        axes[x].set_xlim(Bounds[x])

    plt.setp(axes, yticks=[])
    f.canvas.set_window_title(f'Calibrated parameter ranges with {NbRuns} samples')
    plt.tight_layout()
    plt.show()
