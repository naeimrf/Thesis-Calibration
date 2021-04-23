import os, time, numpy as np, pandas as pd
import local_setup as setup
import local_utility as lu
import local_builder as lb
import matplotlib.pyplot as plt
from SALib.analyze import morris, rbd_fast
from SALib.plotting.morris import horizontal_bar_plot, covariance_plot, sample_histograms
from skopt.space import Space


start_time = time.time()


# METHODS RELATED TO SENSITIVITY ANALYSIS  - - - - - - - - - - - -
def prepare_sensitivity_requirements(name, bounds, inputs, outputs):
    """
    This method retrieves random-numbers generated for simulation cases as well as the results
    from simulation runs. The return values are ready to pass to sensitivity analysis methods in SALib.
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


def run_morris_analysis(problem, x, y, built_in_plots=True):
    """
    Results are:
    mu - the mean elementary effect
    mu_star - the absolute of the mean elementary effect
    sigma - the standard deviation of the elementary effect
    mu_star_conf - the bootstrapped confidence interval
    """
    print(f'\t- The start of Morris sensitivity analysis method ...')
    # https://salib.readthedocs.io/en/latest/_modules/SALib/analyze/morris.html
    Si = morris.analyze(problem, x, y, conf_level=0.95, print_to_console=True, num_levels=4, num_resamples=1000)

    if built_in_plots:
        # ‘xx-small’, ‘x-small’, ‘small’, ‘medium’, ‘large’, ‘x-large’, ‘xx-large’
        params = {'legend.fontsize': 'large',
                  'figure.figsize': (11, 4),
                  'axes.labelsize': 'medium',
                  'axes.titlesize': 'medium',
                  'xtick.labelsize': 'small',
                  'ytick.labelsize': 'small'}
        plt.rcParams.update(params)

        fig1, (ax1, ax2) = plt.subplots(1, 2)  # , figsize=(12, 3)
        fig1.canvas.set_window_title(f'Morris with the total effects of the input factors (μ*)')
        horizontal_bar_plot(ax1, Si, {}, sortby='mu_star', unit=r'')
        covariance_plot(ax2, Si, {}, unit=r"")

        fig2 = plt.figure()  # figsize=(12, 3)
        fig2.canvas.set_window_title(f'Histograms of the input samples for different parameters!')
        fig2.text(0.06, 0.5, 'Number of occurrence', va='center', rotation='vertical')
        sample_histograms(fig2, x, problem, {'color': 'dodgerblue'})
        plt.show()

    labels = Si['names']
    mu_scores = np.array(Si['mu'])
    mu_scores[mu_scores < 0] = 0.001  # replace negative sensitivity values with 0.001

    fig3, ax3 = plt.subplots(figsize=(7, 3))
    y_pos = np.arange(len(labels))
    color = ['darkkhaki' if (xx < max(mu_scores)) else 'limegreen' for xx in mu_scores]
    ax3.barh(y_pos, mu_scores, align='center', color=color, height=0.7)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(labels, size=9)
    ax3.invert_yaxis()  # labels read top-to-bottom
    ax3.set_xlabel('Sensitivity Index')

    fig3.canvas.set_window_title(f'Morris with the primary effect of the input factors (μ)')
    plt.tight_layout()
    plt.show()

    # all measures in one plot!
    df = pd.DataFrame(Si, index=labels)
    df.plot(kind='bar', alpha=0.75, rot=0)
    plt.xlabel("Parameters")
    plt.ylabel("Sensitivity Index")
    fig3 = plt.gcf()
    fig3.canvas.set_window_title(f'Morris method with all measures')
    plt.tight_layout()
    plt.show()
    print(f'\t+ Morris sensitivity analysis method is over!')


def run_rbd_fast_analysis(problem, x, y):
    """
    Based ob SALib package
    Result is a dictionary with keys ‘S1’, where each entry is a list of size D
    (the number of parameters) containing the indices in the same order as the parameter file.
    """
    print(f'\t- The start of RDB FAST sensitivity analysis method ...')
    Si = rbd_fast.analyze(problem, x, y, print_to_console=True)

    labels = Si['names']
    sizes = np.array(Si['S1'])
    sizes[sizes < 0] = 0.001  # replace negative sensitivity values with 0.001

    fig1, ax1 = plt.subplots(figsize=(7, 3))
    """
    explode = np.zeros(len(labels))
    idx = np.where(sizes == np.max(sizes))
    explode[idx] = 0.02
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            normalize=True, shadow=False, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    """
    y_pos = np.arange(len(labels))
    color = ['slategrey' if (xx < max(sizes)) else 'limegreen' for xx in sizes]
    ax1.barh(y_pos, sizes, align='center', color=color, height=0.7)
    for y, x in zip(y_pos, sizes):
        plt.annotate(str(round(x, 3)), xy=(x, y), va='center', size=8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, size=9)
    ax1.invert_yaxis()  # labels read top-to-bottom
    ax1.set_xlabel('Sensitivity Index')
    # ax1.set_title('RDB')

    print(f'\t+ RDB FAST sensitivity analysis method is over!')
    fig1.canvas.set_window_title(f'RDB-FAST: First order of input factors')
    plt.tight_layout()
    plt.show()


# SENSITIVITY RUN * * * * * * * *
print(f"* Sensitivity analysis started ...")
test_path = os.path.join(os.getcwd(), setup.CaseName)
os.chdir(test_path)
print(f'test_path:{test_path}')
res_path = test_path + "/Sim_Results"


_, _, _, _, _, _, params_build = lu.read_epc_values(setup.VarName2Change, 0, res_path)
sim_data, _ = lu.read_simulation_files(res_path)
total_sim_results = lu.get_simulation_runs(setup.BuildNum, sim_data)
problem, x, y = prepare_sensitivity_requirements(setup.VarName2Change, setup.Bounds, params_build, total_sim_results)
try:
    run_morris_analysis(problem, x, y, built_in_plots=False)
except:
    msg = "\t-> Number of samples in model output file must be a multiple of (D+1),\n" \
          "\t   where D is the number of parameters (or groups) in your parameter file."
    print(f'\t-> Morris requirements/restrictions:\n{msg}')
run_rbd_fast_analysis(problem, x, y)

print(f"* Execution time: {round((time.time() - start_time), 2)}s /"
      f" {round(((time.time() - start_time) / 60), 2)}min!")

# ** EXPERIMENTAL CODE **
space = Space([(0.15, float(0.35))])
# centered, salib = lp.generate_randomness(space.dimensions, setup.sample_nbr, space)
# lp.compare2_methods(centered, salib, space.dimensions, setup.sample_nbr, 'Centered-LHC', 'SALib-LHC')
