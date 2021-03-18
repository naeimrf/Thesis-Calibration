from local_calib import *
import matplotlib.pyplot as plt
from SALib.analyze import morris, rbd_fast
from SALib.plotting.morris import horizontal_bar_plot, covariance_plot, sample_histograms
from SALib.sample import saltelli
from SALib.analyze import sobol

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
    Si = morris.analyze(problem, x, y, print_to_console=True, num_levels=10, num_resamples=100)

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig1.canvas.set_window_title(f'MORRIS > the total effects of the input factors!')
    horizontal_bar_plot(ax1, Si, {}, sortby='mu_star', unit=r'')
    covariance_plot(ax2, Si, {}, unit=r"")

    # fig2 = plt.figure(figsize=(10, 4))
    # fig2.canvas.set_window_title(f'Histograms of the input samples for different parameters!')
    # sample_histograms(fig2, x, problem, {'color': 'dodgerblue'})

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
_, _, _, _, _, _, params_build = read_epc_values(lb.VarName2Change, 0)
sim_data, _ = read_simulation_files()
total_sim_results = get_simulation_runs(lb.BuildNum, sim_data)
problem, x, y = prepare_sensitivity_requirements(lb.VarName2Change, lb.Bounds, params_build, total_sim_results)
run_morris_analysis(problem, x, y)
run_rbd_fast_analysis(problem, x, y)

print(f"* Execution time:{round((time.time() - start_time), 2)}s /"
      f" {round(((time.time() - start_time) / 60), 2)}min!")


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# TODO: A quick and small comparison of random generation methods
from skopt.sampler import Sobol, Lhs
from skopt.space import Space


def generate_randomness(dimensions, nbr_samples):
    """
    The purpose of this method is to check the randomness of different algorithms, such as:
    Sobol, different flavors of LHC, MCMC, Hammersly, Morris and Halton but under a reasonably low number of sampling
    """

    fig, ax = plt.subplots(nrows=4, ncols=1)
    ref = 0

    plt.subplot(4, 1, 1)
    sobol = Sobol()
    sb = sobol.generate(dimensions, nbr_samples)
    plt.plot(sb, np.zeros_like(sb) + ref, 'ko', label='Sobol')
    plt.title("Sobol")
    plt.grid()

    plt.subplot(4, 1, 2)
    lhs = Lhs(lhs_type="classic", criterion=None)
    classic = lhs.generate(space.dimensions, nbr_samples)
    plt.plot(classic, np.zeros_like(classic) + ref, 'ro', label='classic lhc')
    plt.title("Classic Latin hypercube")
    plt.grid()

    plt.subplot(4, 1, 3)

    plt.subplot(4, 1, 4)

    plt.grid()
    plt.show()


# space = Space([(0., 1.)])
# generate_randomness(space.dimensions, 10)
