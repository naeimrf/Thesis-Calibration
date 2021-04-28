import time, pickle, glob, shutil, json
from local_setup import *
import local_setup as ls
import local_utility as lu
import local_plots as lp
from local_builder import LocalBuilder
import numpy as np, seaborn as sns, pandas as pd
import matplotlib.pyplot as plt
import math, random
from scipy import stats, linalg
from SALib.sample import latin

start_time = time.time()
test_path = os.path.join(os.getcwd(), CaseName)

# 'Sim_Results' is added with respect to LaunchSim.py line 27
res_path = test_path + "/Sim_Results"
# Change the current working directory to the result folder
if not RECURSIVE_CALIBRATION:
    os.chdir(test_path)


def compare_results(buildings, simulations, measurements, area,
                    alpha=5, per_building=False, all_buildings=True, plots=True):
    """
    This method compares measurements and EPC values with respect to heating and electric load!
    The method returns total energy both for simulation and measurement, their difference in terms of error
    which is an absolute value and the list of acceptable buildings with an error value lower than alpha value.
    """
    total_sim_result = lu.get_simulation_runs(buildings, simulations)

    # - - - - - - - - - - - - - - - -
    reference = {}  # from measurements
    for nbr in buildings:
        tmp = measurements[nbr]['Heating'] + measurements[nbr]['ElecLoad']
        reference.update({nbr: tmp})

    errors = {}
    for nbr in buildings:
        ep_area = area[nbr]['EPlusHeatArea']
        errors[nbr] = {}
        for key, value in total_sim_result[nbr].items():
            err = abs((reference[nbr] / ep_area - value / ep_area) / (reference[nbr] / ep_area)) * 100
            errors[nbr].update({key: err})

        if plots and per_building:
            lp.passed_cases_one_building(errors, nbr, alpha)

    if plots and all_buildings:
        lp.passed_cases_all_buildings(errors, alpha)

    # KEEP acceptable buildings with respect to alpha!
    acceptable = {}
    for nbr in buildings:
        temp = dict((k1, v1) for k1, v1 in errors[nbr].items() if v1 <= alpha)
        acceptable.update({nbr: temp})

    return total_sim_result, reference, errors, acceptable


def get_correlation_pearsonr(x1, y1, **kws):
    (r, p) = stats.pearsonr(x1, y1)
    ax = plt.gca()
    ax.annotate("Pearsonr={:.2f}".format(r), xy=(.1, .9), xycoords=ax.transAxes)
    # ax.annotate("Pearsonp={:.3f}".format(p), xy=(.1, .8), xycoords=ax.transAxes)


def get_correlation_spearmanr(x, y, **kwds):
    # https://stackoverflow.com/questions/66108908/how-to-combine-a-pairplot-and-a-triangular-heatmap
    cmap = kwds['cmap']
    norm = kwds['norm']
    ax = plt.gca()
    ax.tick_params(bottom=False, top=False, left=False, right=False)
    sns.despine(ax=ax, bottom=True, top=True, left=True, right=True)
    r, _ = stats.spearmanr(x, y)
    facecolor = cmap(norm(r))
    ax.set_facecolor(facecolor)
    lightness = (max(facecolor[:3]) + min(facecolor[:3])) / 2
    ax.annotate(f"Spearmanr={r:.2f}", xy=(.1, .9), xycoords=ax.transAxes,
                color='white' if lightness < 0.7 else 'black', size=9)  # , ha='center', va='center'


def make_joint_distribution(buildings, acceptable, res_path, discrete, plot=True):
    print(f'\t- The start of Joint distribution method ...')
    os.chdir(res_path)
    accepted_ranges = {'Buildings': []}
    for b in buildings:
        to_open = list(acceptable[b].keys())
        for tmp in to_open:
            with open(tmp + '.pickle', 'rb') as handle:
                case = pickle.load(handle)
            case_BuildDB = case['BuildDB']

            for param in VarName2Change:
                if param not in accepted_ranges.keys():
                    accepted_ranges[param] = []

                p = getattr(case_BuildDB, param)
                accepted_ranges[param].append(p)
            accepted_ranges['Buildings'].append(b)

    df = pd.DataFrame(accepted_ranges)

    if plot:
        # https://seaborn.pydata.org/tutorial/color_palettes.html
        df_tmp = df.copy()
        df_tmp.loc[:, 'Buildings'] = 'Explained'
        g = sns.pairplot(df_tmp, hue="Buildings", palette="Set2", diag_kind="kde", height=1.5)
        g.map_diag(sns.histplot, hue=None, color=".3", bins=discrete)
        # g.map_offdiag(sns.regplot)
        g.map_upper(sns.regplot)
        g.map_lower(sns.kdeplot, levels=discrete, color="k", shade=True)
        g.map_lower(get_correlation_spearmanr, cmap=plt.get_cmap('vlag'), norm=plt.Normalize(vmin=-.5, vmax=.5))
        fig = plt.gcf()
        fig.canvas.set_window_title(f'Possible correlations among {df_tmp.shape[0]}'
                                    f' validated combination of parameters')
        g.fig.set_size_inches(9, 6)
        g.map_upper(get_correlation_pearsonr)
        plt.show()

    print(f'\t+ Joint distribution method is over!')
    return df


def make_random_from_correlated(data, var_names, nbr_samples, cholesky=True, plots=True):
    """
    # 'eigenvectors' -> cholesky=False
    This function generates random samples from a covariance matrix based on correlated samples
    or directly from correlated samples to calculate covariance matrix first.
    Explanation below from:
    https://scipy-cookbook.readthedocs.io/items/CorrelatedRandomSamples.html

    "To generate correlated normally distributed random samples, one can first generate uncorrelated
    samples, and then multiply them by a matrix C such that CCT=R, where R is the desired covariance
    matrix. C can be created, for example, by using the Cholesky decomposition of R, or from the
    eigenvalues and eigenvectors of R."
    """
    print(f'\t- The start of generating calibrated samples from correlated parameters ...')
    mrfc_time = time.time()
    if isinstance(data, np.ndarray):
        values = data
        covariance_matrix = np.cov(values, rowvar=True)  # The desired covariance matrix
    else:
        # from dataframe columns (joint distribution)
        values = data.to_numpy()
        var_names = list(data.columns.values)
        covariance_matrix = np.cov(values, rowvar=False)  # The desired covariance matrix

    # print(f'covariance_matrix:\n{covariance_matrix}')
    # Generate samples from all independent normally distributed random variables
    # (with mean 0 and std. dev. 1)
    x = stats.norm.rvs(size=(len(var_names), nbr_samples))

    if cholesky:
        # Compute the Cholesky decomposition.
        c = linalg.cholesky(covariance_matrix, lower=True)
    else:
        # Compute the eigenvalues and eigenvectors.
        evals, evecs = linalg.eigh(covariance_matrix)
        # Construct c, so c*c^T = r.
        c = np.dot(evecs, np.diag(np.sqrt(evals)))

    # Convert the data to correlated random variables
    y = np.dot(c, x)

    y_transformed = []
    if not isinstance(data, np.ndarray):
        # transform data to the correct range of calibrated values
        for i, j in zip(range(len(y)), range(len(values[0]))):
            # one-dimensional piecewise linear interpolant to a function with given discrete data points
            # https://numpy.org/doc/stable/reference/generated/numpy.interp.html
            y_transformed.append(np.interp(y[i], (y[i].min(), y[i].max()), (values[:, j].min(), values[:, j].max())))
        y_transformed = np.asarray(y_transformed, dtype=np.float32)
    else:
        y_transformed = y

    if plots:
        combs = []
        combinations_names = []
        for i in range(len(var_names)):
            for j in range(len(var_names)):
                if i > j:
                    combs.append((i, j))
                    combinations_names.append((var_names[i], var_names[j]))

        plots_mtx = list(range(1, (len(var_names) - 1) * (len(var_names) - 1) + 1))
        counter = 0
        for i in range(len(var_names) - 1):
            for j in range(len(var_names) - 1):
                counter += 1
                if i < j:
                    plots_mtx.remove(counter)

        for i, j, k in zip(plots_mtx, combs, combinations_names):
            plt.subplot(len(var_names) - 1, len(var_names) - 1, i)
            plt.scatter(y_transformed[j[0]], y_transformed[j[1]], marker='h', c='blue')
            plt.xlabel(k[0])
            plt.ylabel(k[1])
            plt.tight_layout()
            plt.grid(False)

        fig = plt.gcf()
        fig.canvas.set_window_title(
            f"Pairwise plots of {nbr_samples} generated samples from covariance matrix of calibrated parameters")
        plt.show()

    tmp = np.array(y_transformed[0]).reshape((-1, 1))  # TRANSPOSE TO COLUMN FORMAT
    for i in range(len(y_transformed) - 1):
        col_tmp = (y_transformed[i + 1]).reshape(-1, 1)
        tmp = np.concatenate((tmp, col_tmp), axis=1)

    y_transformed = tmp
    print(
        f'\t+ Generating calibrated samples from correlated parameters is done in {round(time.time() - mrfc_time, 2)}s.')
    return y_transformed


def fit_best_distro(calib_params, nbr_samples, nbr_match, Danoe=True, plot=True):
    """
    The third parameter is used when plot option is set to 'True' for presenting the number of distributions
    This code is inspired by Sebastian Jose's answer in stackoverflow
    # https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python
    """
    read_time3 = time.time()
    print(f'\t- Finding the best distributions started ...')

    # distributions = ['foldcauchy', 'cauchy', 'alpha', 'dweibull', 'genextreme', 'pearson3', 'dgamma']
    distributions = ['alpha', 'anglit', 'arcsine', 'beta', 'betaprime', 'bradford', 'burr', 'cauchy', 'chi', 'chi2',
                     'cosine', 'dgamma', 'dweibull', 'erlang', 'expon', 'exponweib', 'exponpow', 'f', 'fatiguelife',
                     'fisk', 'foldcauchy', 'foldnorm', 'genlogistic', 'genpareto', 'genexpon', 'gumbel_r',
                     'genextreme', 'gausshyper', 'gamma', 'gengamma', 'genhalflogistic', 'gilbrat', 'gompertz',
                     'gumbel_l', 'halfcauchy', 'halflogistic', 'halfnorm', 'hypsecant', 'invgamma', 'invgauss',
                     'invweibull', 'johnsonsb', 'johnsonsu', 'ksone', 'kstwobign', 'laplace', 'logistic', 'loggamma',
                     'loglaplace', 'lognorm', 'lomax', 'maxwell', 'mielke', 'nakagami', 'ncx2', 'ncf', 'nct', 'norm',
                     'pareto', 'pearson3', 'powerlaw', 'powerlognorm', 'powernorm', 'rdist', 'reciprocal', 'rayleigh',
                     'rice', 'recipinvgauss', 'semicircular', 't', 'triang', 'truncexpon', 'truncnorm', 'tukeylambda',
                     'uniform', 'vonmises', 'wald', 'weibull_min', 'weibull_max', 'wrapcauchy']

    param_names = list(calib_params.columns.values)
    # Danoe's formula to find optimum number of bins
    if Danoe:
        n = len(calib_params)
        num_bins = []
        skewness = stats.skew(calib_params, axis=0)  # skewness along column-wise
        sigma_g1 = math.sqrt((6 * (n - 2)) / ((n + 1) * (n + 3)))
        for skew in skewness:
            num_bins.append(round(1 + math.log(n, 2) + math.log(1 + abs(skew) / sigma_g1, 2)))
    else:
        # predefined number of bins for each parameter in local_setup
        num_bins = [nbr_samples] * len(calib_params.columns)

    # Calculate Histogram
    frequencies = []
    bin_edges = []
    central_values = []
    for name, nb in zip(param_names, range(len(num_bins))):
        f, b = np.histogram(calib_params[name], num_bins[nb], density=True)
        frequencies.append(f)
        bin_edges.append(b)
        central_values.append([(bin_edges[nb][i] + bin_edges[nb][i + 1]) / 2 for i in range(len(bin_edges[nb]) - 1)])

    results = {}
    best_match = {}
    for i, name in enumerate(param_names):
        results[name] = {}
        best_match[name] = {}
        for distribution in distributions:
            dist = getattr(stats, distribution)

            fitted = dist.fit(calib_params[name])  # Get parameters of distribution
            # Separate parts of parameters
            arg = fitted[:-2]
            loc = fitted[-2]
            scale = fitted[-1]

            # Calculate fitted PDF and error with fit in distribution
            pdf_values = [dist.pdf(c, loc=loc, scale=scale, *arg) for c in central_values[i]]

            # Calculate SSE (sum of squared estimate of errors)
            sse = np.sum(np.power(frequencies[i] - pdf_values, 2.0))

            # Build results and sort by sse
            results[name][distribution] = [sse, arg, loc, scale]
        results[name] = {k: results[name][k] for k in sorted(results[name], key=results[name].get)}
        first_key = next(iter(results[name]))
        best_match[name].update({first_key: next(iter(results[name].values()))})
    print(f'\t+ Finding the best distributions is done in {round((time.time() - read_time3), 2)}s!')

    if plot:
        for key in results.keys():
            distro_to_plot = {k: results[key][k] for k in list(results[key])[:nbr_match]}

            # Histogram of data
            plt.figure(figsize=(6, 4))
            plt.hist(calib_params[key], density=True, ec='white', color='gray')
            plt.title(key)
            plt.xlabel('Values')
            plt.ylabel('Frequencies')

            legends = []
            # Plot n distributions
            for distro, result in distro_to_plot.items():
                legends.append(distro)
                dist = getattr(stats, distro)
                sse = result[0]
                arg = result[1]
                loc = result[2]
                scale = result[3]
                x_plot = np.linspace(min(calib_params[key]), max(calib_params[key]), 10)
                y_plot = dist.pdf(x_plot, loc=loc, scale=scale, *arg)

                # FIXME: A QUICK SOLUTION TO AVOID ERROR JUMP IN y_plot!
                if lu.second_largest(y_plot) * 10 < max(y_plot):
                    index_max = max(range(len(y_plot)), key=y_plot.__getitem__)
                    y_plot[index_max] = lu.second_largest(y_plot) * 2

                plt.plot(x_plot, y_plot, label=distro + ": " + str(sse)[0:6],
                         color=(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))

            plt.legend(title='Sum of squared of errors', bbox_to_anchor=(0, 1), loc='upper left')
            fig = plt.gcf()
            fig.canvas.set_window_title("Best theoretical distribution fits to calibrated parameters")
            plt.show()

    return best_match


def make_random_from_continuous(distro_dict, bounds, nbr_samples, plot):
    print(f'\t- The start of generating calibrated samples from the best distribution ...')
    mrfc_tme = time.time()
    param_names = list(distro_dict.keys())
    distro_names = []
    for param in param_names:
        distro_names += distro_dict[param].keys()

    problem = {
        'num_vars': len(param_names),
        'names': param_names,
        'bounds': [[0, 1]] * len(param_names)
    }
    salib_samples = latin.sample(problem, int(nbr_samples * 2))  # double size samples to trim later
    for i, dist in zip(range(len(param_names)), distro_names):
        distro = getattr(stats, dist)

        distro_values = (distro_dict[param_names[i]])[dist]
        arg = distro_values[1]
        loc = distro_values[2]
        scale = distro_values[3]

        salib_samples[:, i] = distro.ppf(salib_samples[:, i], loc=loc, scale=scale, *arg)

    # FIXME: A QUICK SOLUTION TO AVOID OUT OF BOUNDS IN SALib_SAMPLES, TRIM 100 SAMPLES!
    tmp = []
    for i, bound in enumerate(bounds):
        tmp.append((salib_samples[:, i])[np.logical_not(np.logical_or(
            salib_samples[:, i] > bound[1], salib_samples[:, i] < bound[0]))])

    salib_samples = np.array((tmp[0][:nbr_samples])).reshape((-1, 1))  # TRANSPOSE TO COLUMN FORMAT
    for i in range(len(tmp) - 1):
        col_tmp = (tmp[i + 1][:nbr_samples]).reshape(-1, 1)
        salib_samples = np.concatenate((salib_samples, col_tmp), axis=1)

    print(f'\t+ Generating calibrated samples from best distribution is done in {round(time.time() - mrfc_tme, 2)}s.')
    df = pd.DataFrame(salib_samples, columns=param_names)
    if plot:
        sns.color_palette("rocket", as_cmap=True)
        g = sns.PairGrid(df, palette="Paired")
        g.map_diag(sns.histplot, kde=True)  # element="step"
        g.map_lower(sns.kdeplot, levels=4)
        g.map_lower(sns.scatterplot, color="gray")
        g.map_upper(sns.regplot,
                    scatter_kws={"color": "gray"}, line_kws={"color": "red"})
        g.add_legend(frameon=True)
        fig = plt.gcf()
        fig.canvas.set_window_title(f'Random samples for parameters from calibrated ranges')
        g.fig.set_size_inches(9, 6)
        plt.tight_layout()
        plt.show()

    else:
        for param_name in param_names:
            print(f'{param_name} -> min:{np.min(df[param_name])}, max:{np.max(df[param_name])}')

    return salib_samples


def test_of_assumption_beta(buildings, acceptable, beta):
    unexplained_buildings = []
    ratio = 0
    for b in buildings:
        if len(acceptable[b]) >= 1:
            ratio += 1
        else:
            unexplained_buildings.append(b)

    percentage = (ratio / len(buildings)) * 100
    if percentage > beta:
        if not RECURSIVE_CALIBRATION:
            print(f"\t-> β qualification satisfied with {round(percentage, 2)}%.")
        return False, []
    else:
        if not RECURSIVE_CALIBRATION:
            print(f"\t-> ATTENTION: only {round(percentage, 2)}% simulations of buildings matched measurements!\n"
                  # f"\t-> Model revision, choice of θ parameters or more number of simulations is needed!\n"
                  f"\t\t-> List of unexplained buildings: {unexplained_buildings}")
        return True, unexplained_buildings


def calibrate_uncertain_params(params, params_ranges, nbr_sim_uncalib, buildings, final_samples=100,
                               alpha=5, beta=85, discrete=5, all_plots=True, approach=1):
    """
    This function is based on Carlos Cerezo, et al's method published in 2017
    This method needs:
    'params' -> uncertain parameter names
    'params_ranges' -> uncertain parameter ranges
    'nbr_sim' ->  the number simulated files in local_builder.py
    'buildings' -> list of building names simulated in local_builder
    'alpha' value -> the upper acceptable limit for simulation and measurement difference for each building
    'beta' value -> minimum percentage of acceptable simulations which satisfies 'alpha' condition
    'discrete' -> as the number of division for continuous parameter ranges
    'approach' -> to create random samples from calibrated parameters, 1 for covariance matrix and 2
    for best theoretical distribution fit and LHC sampling of that distribution.
    'final_samples' -> number of random samples from joint distribution
    """

    print(f'- The start of annual calibration method with alpha: {alpha} ...')
    if not all_plots:
        print("** THE PLOT OPTION IS OFF, CHANGE TO 'TRUE' TO SEE THE RESULTS! **")
    # * ALGORITHM STEP1: PARAMETER DEFINITION * * * * * * * * * * /
    _, a_temp, total_epc, _, _, params_cat, params_build = lu.read_epc_values(VarName2Change, 0, res_path)
    if all_plots:
        lp.plot_prior_distributions(params, params_ranges, buildings, discrete,
                                    per_building=False, SALib=SAMPLE_TYPE, LHC=params_cat)

    # * ALGORITHM STEP2: PARAMETRIC SIMULATION * * * * * * * * * * /
    print(f"\t-> {nbr_sim_uncalib} random simulations out of {discrete ** (len(params))} possible combinations!")
    sim_data, model_area = lu.read_simulation_files(res_path)

    # * ALGORITHM STEP3: ERROR QUALIFICATION (α) * * * * * * * * * * /
    total_sim_results, _, errors, acceptable = \
        compare_results(buildings, sim_data, total_epc, model_area, alpha=alpha,
                        per_building=True, all_buildings=True, plots=all_plots)

    # * ALGORITHM STEP4: TEST OF ASSUMPTIONS (β) * * * * * * * * * * /
    low_beta, _ = test_of_assumption_beta(buildings, acceptable, beta)
    if low_beta:
        return 1

    # * ALGORITHM STEP5: DISTRIBUTION GENERATION * * * * * * * * * * /
    calib_params = make_joint_distribution(buildings, acceptable, res_path, discrete, plot=all_plots)

    if all_plots:
        lp.make_t_SNE_plot(calib_params)

    # Plot below provides more insight to possible correlations of parameters /
    # if all_plots: # FIXME: AN EXTRA PLOT, CORRELATION OF PARAMETERS, UNCOMMENT TO SEE.
    #    plot_joint_distributions(calib_params, discrete)

    calib_frequencies = lp.plot_calibrated_parameters(calib_params, Bounds, discrete, NbRuns, alpha, plot=all_plots)

    # * ALGORITHM STEP6: RANDOM SAMPLED SIMULATIONS * * * * * * * * * * /
    # Save and send the result back to local_builder for simulations with θ'
    smpl = final_samples  # The number of final samples based on calibrated parameters

    if approach == 1:
        theta_prime = make_random_from_correlated(calib_params, VarName2Change, smpl,
                                                  cholesky=True, plots=all_plots)
        np.savetxt("theta_prime.csv", theta_prime, delimiter=",")

    elif approach == 2:
        best_distros = fit_best_distro(calib_params, sample_nbr, 3, Danoe=False, plot=all_plots)
        theta_prime = make_random_from_continuous(best_distros, Bounds, smpl, plot=all_plots)
        np.savetxt("theta_prime.csv", theta_prime, delimiter=",")

    print(f'\t-> {smpl} calibrated parameters are saved as "theta_prime.csv" file in:\n\t{res_path}')
    print(f'+ Calibration method is over!')
    return 0


def return_best_combination_for_each_building(path, params_build, errors, param_names):
    best_simulations = {}
    building_ids = errors.keys()
    for id in building_ids:
        best_version = min(errors[id].items(), key=lambda x: x[1])
        best_simulations.update({id: best_version[0]})

    best_combinations = {}
    for id, version in zip(building_ids, best_simulations.values()):
        best_combinations.update({id: params_build[id][version]})

    df = pd.DataFrame(best_combinations, index=param_names)
    print(f'\t-> Best combination of probabilistic parameters for buildings:\n{df}')

    df.to_csv('best_combinations.csv', sep=',', index=True)

    print(f"\t-> Best combinations of probabilistic parameters for all simulated "
          f"buildings are saved as 'best_combinations.csv' file in:\n\t-> {path}")
    return df


def recursive_calibration(buildings, CaseName, alpha=5, beta=85, iterations=2):
    """
    This function works if RECURSIVE_CALIBRATION is set to 'True'
    It is an attempt to find calibrated parameters with less computational resources.
    """
    parent_folder = os.getcwd()
    if os.path.exists('./CSV/'):
        files = glob.glob('./CSV/*')
        for f in files:
            os.remove(f)
        os.rmdir('./CSV/')
        print("* Previous CSV file deleted *")
    os.mkdir('./CSV/')

    iterations_best_results = {}
    for iteration in range(iterations):

        builder = LocalBuilder()
        if iteration > 0:
            # First simulation run to get uncalibrated parameters
            # After the first run we read calibrated parameters from previous run
            os.chdir(parent_folder)
            ls.CALIBRATE_WITH_CALIBRATED_PARAMETERS = True
        builder.run(CaseName)
        os.chdir(test_path)

        _, a_temp, total_epc, _, _, params_cat, params_build = lu.read_epc_values(VarName2Change, 0, res_path)
        sim_data, model_area = lu.read_simulation_files(res_path)
        total_sim_results, _, errors, acceptable = \
            compare_results(buildings, sim_data, total_epc, model_area, alpha=alpha,
                            per_building=True, all_buildings=True, plots=False)
        for_if = for_elif = 1
        while True:
            low_beta, unexplained_buildings = test_of_assumption_beta(buildings, acceptable, beta)

            # we need at least p data points in p dimensions where the dimension is the number of parameters
            # print(f"\t-> Acceptable number of simulations: {sum(len(v) for v in acceptable.values())}, low_β: {low_beta}")
            if low_beta:
                alpha_tmp = alpha + for_if

                for nbr in buildings:
                    if nbr in unexplained_buildings:
                        temp = dict((k1, v1) for k1, v1 in errors[nbr].items() if v1 <= alpha_tmp)
                        if temp:
                            print(f"\t-> Alpha value for building {nbr} sets temporarily to:{alpha_tmp}")
                            acceptable.update({nbr: temp})

                for_if += 1

            elif sum(len(v) for v in acceptable.values()) < len(VarName2Change):
                alpha_tmp = alpha + for_elif
                for nbr in buildings:
                    temp = dict((k1, v1) for k1, v1 in errors[nbr].items() if v1 <= alpha_tmp)
                    if temp:
                        acceptable.update({nbr: temp})
                for_elif += 1
            else:
                break

        print(f">> {acceptable}")
        iterations_best_results[iteration] = {}
        for key, value in acceptable.items():
            iterations_best_results[iteration].update({key: min(acceptable[key].values())})

        calib_params = make_joint_distribution(buildings, acceptable, res_path, 5, plot=False)
        theta_prime = make_random_from_correlated(calib_params, VarName2Change, NbRuns,
                                                  cholesky=True, plots=False)

        csv_file = "theta_prime.csv"
        path_to_save = parent_folder + "/CSV/" + csv_file
        if os.path.exists(path_to_save):
            os.remove(path_to_save)

        np.savetxt(path_to_save, theta_prime, delimiter=",")
        print(f'\t-> {iterations}: {NbRuns} calibrated parameters are saved as {csv_file} in:\n\t{path_to_save}')

        converge = []
        for value in iterations_best_results.values():
            converge.append(list(value.values()))
        converge = np.array([j for sub in converge for j in sub])
        if all(converge < alpha):
            with open('converge.txt', 'w') as file:
                file.write(json.dumps(iterations_best_results)) # use `json.loads` to do the reverse
            return iterations_best_results

        os.chdir(parent_folder)
        if os.path.exists(CaseName):
            shutil.rmtree(parent_folder + "/" + CaseName)


# CALIBRATION RUN * * * * * * * *
if __name__ == '__main__':
    if RECURSIVE_CALIBRATION:
        best_results = recursive_calibration(BuildNum, CaseName, alpha=15, beta=85, iterations=3)
        lp.plot_recursive_improvement(best_results)
    else:
        if CALIBRATE_WITH_CALIBRATED_PARAMETERS:
            print(f'* Simulations for buildings: {BuildNum} with calibrated parameters *')
            _, a_temp, total_epc, _, _, params_cat, params_build = lu.read_epc_values(VarName2Change, 0, res_path)
            sim_data, model_area = lu.read_simulation_files(res_path)
            total_sim_results, _, errors, acceptable = \
                compare_results(BuildNum, sim_data, total_epc, model_area, alpha=5,
                                per_building=False, all_buildings=True, plots=True)
            _ = return_best_combination_for_each_building(res_path, params_build, errors, VarName2Change)

            # TODO: BELOW ARE DUMMY VALUES, FIX IT!
            n = 1000
            u1 = 5
            u2 = 5
            series1 = u1 + np.random.randn(n)
            series2 = u2 + np.random.randn(n)
            # lp.plot_metered_vs_simulated_energies(series1, series2, bins=20)
            print(f'** Illustration of the results by calibrated parameters is over *')
        else:
            # alpha=5%, based on ASHRAE Guideline 14–2002
            calibrate_uncertain_params(VarName2Change, Bounds, NbRuns, BuildNum, alpha=5,
                                       beta=90, final_samples=100, discrete=sample_nbr, all_plots=False, approach=1)

            print(f"* Execution time:{round((time.time() - start_time), 2)}s /"
                  f" {round(((time.time() - start_time) / 60), 2)}min!")

            # ** EXPERIMENTAL CODE **
            # Prior presentation of real samples for selected parameters
            # if SAMPLE_TYPE:
            #    combinations, middle_points = lu.all_combinations(Bounds, VarName2Change, sample_nbr)
            #    lp.plot_combinations(VarName2Change, Bounds, combinations, middle_points, sample_nbr)
