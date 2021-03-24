from local_calib import *
import matplotlib.pyplot as plt
from SALib.analyze import morris, rbd_fast
from SALib.sample import latin
from SALib.plotting.morris import horizontal_bar_plot, covariance_plot, sample_histograms
from skopt.space import Space
from skopt.sampler import *

start_time = time.time()


# METHODS RELATED TO SENSITIVITY ANALYSIS  - - - - - - - - - - - -
def prepare_sensitivity_requirements(name, bounds, inputs, outputs):
    """
    This method retrieves random-numbers generated for simulation cases as well as the results
    from simulation runs. The return values are ready to pass sensitivity analysis methods in SALib.
    # https://salib.readthedocs.io/en/latest/
    # https://stackoverflow.com/questions/41045699/performing-a-sensitivity-analysis-with-python
    """
    problem = {'names': name, 'bounds': bounds, 'num_vars': len(name)}

    x = []
    y = []
    for building in inputs.keys():
        for case in inputs[building].keys():
            x.append(inputs[building][case])
            y.append(outputs[building][case])

    x = np.array(x)
    y = np.array(y)
    return problem, x, y


def run_morris_analysis(problem, x, y):
    """
    Results are:
    mu - the mean elementary effect
    mu_star - the absolute of the mean elementary effect
    sigma - the standard deviation of the elementary effect
    mu_star_conf - the bootstrapped confidence interval
    """

    print(f'\t- The start of Morris sensitivity analysis method ...')
    Si = morris.analyze(problem, x, y, print_to_console=True, num_levels=4, num_resamples=100)

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig1.canvas.set_window_title(f'MORRIS > the total effects of the input factors!')
    horizontal_bar_plot(ax1, Si, {}, sortby='mu_star', unit=r'')
    covariance_plot(ax2, Si, {}, unit=r"")

    fig2 = plt.figure(figsize=(10, 4))
    fig2.canvas.set_window_title(f'Histograms of the input samples for different parameters!')
    sample_histograms(fig2, x, problem, {'color': 'dodgerblue'})

    print(f'\t+ Morris sensitivity analysis method is over!')
    plt.show()


def run_rbd_fast_analysis(problem, x, y):
    """
    Based ob SALib package
    Result is a dictionary with keys ‘S1’, where each entry is a list of size D
    (the number of parameters) containing the indices in the same order as the parameter file.
    """
    print(f'\t- The start of RDB FAST sensitivity analysis method ...')
    Si = rbd_fast.analyze(problem, x, y, print_to_console=True)

    labels = Si['names']
    sizes = np.abs(Si['S1'])
    explode = np.zeros(len(labels))
    idx = np.where(sizes == np.max(sizes))
    explode[idx] = 0.05

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            normalize=True, shadow=False, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    print(f'\t+ RDB FAST sensitivity analysis method is over!')
    fig1.canvas.set_window_title(f'RDB-FAST > first order of input factors!')
    plt.show()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# SENSITIVITY RUN * * * * * * * *
print(f"* Sensitivity analysis started ...")
"""
_, _, _, _, _, _, params_build = read_epc_values(lb.VarName2Change, 0)
sim_data, _ = read_simulation_files()
total_sim_results = get_simulation_runs(lb.BuildNum, sim_data)
problem, x, y = prepare_sensitivity_requirements(lb.VarName2Change, lb.Bounds, params_build, total_sim_results)
run_morris_analysis(problem, x, y)
run_rbd_fast_analysis(problem, x, y)
"""
print(f"* Execution time:{round((time.time() - start_time), 2)}s /"
      f" {round(((time.time() - start_time) / 60), 2)}min!")


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
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


def rand_plot_1d(xx, nbr_samples, message, color='b*'):
    plt.plot(xx, np.zeros_like(xx), color)
    plt.title(message, fontsize=10)
    plt.xticks(size=8)
    plt.yticks(size=6)
    for i in np.arange(space.bounds[0][0], space.bounds[0][1],
                       (space.bounds[0][1] - space.bounds[0][0]) / nbr_samples):
        plt.axvline(i, linestyle=':', color='gray')


def generate_randomness(dimensions, nbr_samples):
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
    rand_plot_1d(sb, nbr_samples, 'skopt-Sobol')

    plt.subplot(5, 2, 2)
    lhs = Lhs(lhs_type="classic", criterion=None)
    classic = lhs.generate(space.dimensions, nbr_samples)
    rand_plot_1d(classic, nbr_samples, 'skopt-Classic LHC')

    plt.subplot(5, 2, 3)
    lhs = Lhs(lhs_type="centered", criterion=None)
    centered = lhs.generate(space.dimensions, nbr_samples)
    rand_plot_1d(centered, nbr_samples, 'skopt-Centered LHC', color='gh')

    plt.subplot(5, 2, 4)
    lhs = Lhs(criterion="maximin", iterations=1000)
    maximin = lhs.generate(space.dimensions, nbr_samples)
    rand_plot_1d(maximin, nbr_samples, 'skopt-Maximin LHC')

    plt.subplot(5, 2, 5)
    lhs = Lhs(criterion="correlation", iterations=1000)
    correlation = lhs.generate(space.dimensions, nbr_samples)
    rand_plot_1d(correlation, nbr_samples, 'skopt-Correlation LHC')

    plt.subplot(5, 2, 6)
    lhs = Lhs(criterion="correlation", iterations=1000)
    ratio = lhs.generate(space.dimensions, nbr_samples)
    rand_plot_1d(ratio, nbr_samples, 'skopt-Ratio LHC')

    plt.subplot(5, 2, 7)
    halton = Halton()
    hal = halton.generate(space.dimensions, nbr_samples)
    rand_plot_1d(hal, nbr_samples, 'skopt-Halton')

    plt.subplot(5, 2, 8)
    hammersly = Hammersly()
    ham = hammersly.generate(space.dimensions, nbr_samples)
    rand_plot_1d(ham, nbr_samples, 'skopt-Hammersly')

    plt.subplot(5, 2, 9)
    grid = Grid(border="exclude", use_full_layout=False)
    gr = grid.generate(space.dimensions, nbr_samples)
    rand_plot_1d(gr, nbr_samples, 'skopt-Grid')

    plt.subplot(5, 2, 10)
    problem = {'names': 'x1', 'bounds': [[space.bounds[0][0], space.bounds[0][1]]], 'num_vars': 1}
    salib = latin.sample(problem, nbr_samples)
    rand_plot_1d(salib, nbr_samples, 'SALib-LHC', color='mx')

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

    print(type(dimensions), dimensions[0].bounds[1])
    print(sample_nbr)
    print(f'salib:{salib}')

    nbrs = range(len(flat_data[0]))
    scope = ((dimensions[0].bounds[1] - dimensions[0].bounds[0])/sample_nbr)/2
    plt.errorbar(nbrs, flat_data[0], yerr=scope, ecolor='lightsteelblue', ls='none', capsize=4)
    plt.xticks(nbrs)
    plt.xlabel('Number of samples')
    plt.ylabel('Random value by the method')

    # plt.grid()
    plt.legend()
    fig = plt.gcf()
    fig.canvas.set_window_title('Illustration of samples for two Quasi-Random Number Generators')
    plt.show()


space = Space([(19.5, float(24))])
sample_nbr = 8
centered, salib = generate_randomness(space.dimensions, sample_nbr)
compare2_methods(centered, salib, space.dimensions, sample_nbr, 'Centered-LHC', 'SALib-LHC')
