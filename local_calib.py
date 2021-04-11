import os, time, pickle, copy
import local_builder as lb
import numpy as np, seaborn as sns, pandas as pd
import matplotlib.pyplot as plt
import math, random, statistics
from collections import Counter
from scipy import stats, linalg
from SALib.sample import latin
from pathlib import Path

start_time = time.time()
test_path = os.path.join(os.getcwd(), lb.CaseName)
# print(f'test_path:{test_path}, type:{type(test_path)}')

# 'Sim_Results' is added with respect to LaunchSim.py line 27
res_path = test_path + "/Sim_Results"

# Change the current working directory to the result folder
os.chdir(test_path)


def my_print(*args):
    print("- - - Function results - - -")
    for i in args:
        print(f'-> {i}')
    print("- - - Function Ends - - - - - - -")


def get_file_names(BuildNum, NbRuns, path, alternative):
    """
    This method is to find or make simulation file names
    alternative 0 -> based on generated input file names
    alternative 1 -> based on existing simulation file names
    This function also checks the existing number of files against the number of simulations asked by user!
    """
    if os.path.isfile(path + "/theta_prime.csv"):
        os.remove(path + "/theta_prime.csv")
        print("\t-> Previous calibrated parameter file 'theta_prime.csv' is removed.")

    # check the number of simulations in the result folder and produce their names
    all_files = os.listdir(path)
    if NbRuns * len(BuildNum) != len(all_files) / 2:
        print(f"\t-> Something is WRONG, number of exiting files does not equal to 'BuildNum'!")
        print(f'\t-> Existing result pickle files:{len(all_files) / 2} != Simulations asked in builder.py:{NbRuns * len(BuildNum)}')
        raise ValueError("Missing simulation files!")

    lst = {}
    if alternative:
        # find simulation file names to extract energy results
        # method 1
        for file in all_files:
            if file.endswith('.pickle'):
                for cn in BuildNum:
                    if str(cn) + 'v' in file:  # if file.find(cn):
                        if cn in lst:
                            value = lst[cn]
                            value.append(file)
                        else:
                            lst[cn] = [file]
        # print(f'Names retrieved from existing files:\n{lst}')
    else:
        # recreate simulation file names to extract energy results
        # method 2
        for bld in BuildNum:
            for i in range(lb.NbRuns):
                if bld in lst:
                    value = lst[bld]
                    value.append('Building_' + str(bld) + 'v' + str(i) + ".pickle")
                else:
                    lst[bld] = ['Building_' + str(bld) + 'v' + str(i) + ".pickle"]
        # print(f'Synthetically created files:\n{lst}')
    return lst


def read_epc_values(params_list, surface_type):
    """
    This method reads probabilistic input parameters and EPC values from pickle files in the RESULT folder!
    Select surface_type equal to 1 to apply A_temp or 0 to apply EnergyPLus geometric area!!

    The stored input values in result pickle files retrieved in this function are:
    epc (KWh)-> all non-zero EPC values saved in the result files (pickle format) originally read from a geojson file
    area (m2)-> A_temp and EPHeatedArea areas from geojson file and heated area from Energy plus model
    total (KWh)-> A sum energy consumption for building categorized in groups: 'Heating', 'DHW', 'Cooling', 'ElecLoad' and 'NRJandClass'
    total_per_area (KWh/m2)-> sum of each category of energy normalized by surface area
    enj_per (KWh/m2)-> building energy performance
    prob_params -> Two dictionaries of all probabilistic parameters with their random generated values by Latin Hyper Cube,
    one dictionary to hold probabilistic parameters building-vice and one dictionary to hold them parameter-vice
    """
    read_time1 = time.time()
    print(f'\t- Reading of input files with pickle format started ...')
    input_files = get_file_names(lb.BuildNum, lb.NbRuns, res_path, 1)
    building_ids = input_files.keys()

    epc, area, total, total_per_area, enj_per = dict(), dict(), dict(), dict(), dict()
    prob_params_cat = {}
    prob_params_building = {}
    # 'for-loop' to handle more than one building
    for id in building_ids:
        # only one input_file for each building is enough to get the EPCs!
        temp = open(input_files[id][0], 'rb')
        temp_building = pickle.load(temp)
        temp.close()

        # dig into the BuildData & EPCMeters
        building = temp_building['BuildData']
        epc[id] = copy.deepcopy(building.EPCMeters)

        meters = building.EPCMeters.keys()
        area.update({id: {}})
        area[id].update({'surface': building.surface})
        area[id].update({'EPHeatedArea': building.EPHeatedArea})

        # section below removes zero EPCMeters from the final result!
        # also total energy consumption per surface area, is calculated!
        total[id] = {}
        total_per_area[id] = {}
        for meter in meters:
            tot_tmp = 0
            for key, value in building.EPCMeters[meter].items():
                if not value:
                    del epc[id][meter][key]
                else:
                    if 'NRJ' not in meter:
                        tot_tmp += value
            total[id].update({meter: tot_tmp})

            if surface_type:
                surf = area[id]['surface']
            else:
                surf = area[id]['EPHeatedArea']
            total_per_area[id].update({meter: (tot_tmp / surf)})
        # building energy performance
        enj_per[id] = sum(total_per_area[id].values())

        # retrieve probabilistic parameters for each simulation (building case)!
        prob_params_cat[id] = {}
        prob_params_building[id] = {}
        for i in range(len(input_files[id])):
            build_case = input_files[id][i]
            temp = open(build_case, 'rb')
            temp_building = pickle.load(temp)
            temp.close()

            # the script below gathers parameters CATEGORICAL-vice
            building = temp_building['BuildData']
            for param in params_list:
                if param not in prob_params_cat[id]:
                    prob_params_cat[id][param] = []
                temp_val = getattr(building, param)
                prob_params_cat[id][param].append(temp_val)

                # the script below gathers parameters BUILDING-vice
                building_name = build_case.replace('.pickle', '')
                if building_name not in prob_params_building[id]:
                    prob_params_building[id][building_name] = []
                prob_params_building[id][building_name].append(temp_val)

    print(f'\t+ Reading of inputs is done in {round((time.time() - read_time1), 2)}s!')
    # my_print(epc, area, total, total_per_area, enj_per, prob_params_cat, prob_params_building)
    return epc, area, total, total_per_area, enj_per, prob_params_cat, prob_params_building


def read_simulation_files():
    read_time2 = time.time()
    print(f'\t- Reading of result files with pickle format started ...')
    result_files = get_file_names(lb.BuildNum, lb.NbRuns, res_path, 1)

    os.chdir(res_path)  # change the working directory to result files!

    simulated, model_area = dict(), dict()
    for key, value in result_files.items():
        simulated[key] = {}
        model_area[key] = {}

        for building in result_files[key]:
            building_name = building.replace('.pickle', '')
            simulated[key][building_name] = {}
            with open(building, 'rb') as handle:
                build_result = pickle.load(handle)

            for k, v in build_result['HeatedArea'].items():
                if not isinstance(v[0], str):  # to avoid time-interval name ex. 'Hourly' and units in results!
                    simulated[key][building_name].update({k: (sum(v) / (3600 * 1000))})  # joule to Kwh

        model_area[key].update({'EPlusTotArea': build_result['EPlusTotArea']})
        model_area[key].update({'EPlusHeatArea': build_result['EPlusHeatArea']})
        model_area[key].update({'EPlusNonHeatArea': build_result['EPlusNonHeatArea']})

    print(f'\t+ Reading of result files is done in {round((time.time() - read_time2), 2)}s!')
    # my_print(simulated, model_area)
    return simulated, model_area


def find_closest_value(arr, val):
    arr = np.asarray(arr)
    idx = (np.abs(arr - val)).argmin()
    return arr[idx]


def get_projected_freq(theoretical_xlabels, lpc_xlabels, groups):
    """
    This function fits random numbers from LHC to theoretical distribution range
    and returns the frequencies of projected values from actual LHC values
    """
    lpc_theoretical_projected = []
    for theo_v, lpc_v in zip(theoretical_xlabels, lpc_xlabels):
        tmp = []
        for i in range(len(lpc_v)):
            tmp.append(find_closest_value(theo_v, lpc_v[i]))
        lpc_theoretical_projected.append(tmp)

    lpc_freq = [[] for _ in range(groups)]
    lpc_theoretical_projected = [sorted(i) for i in lpc_theoretical_projected]
    length = len(lpc_theoretical_projected[0])
    j = 0
    for t, l in zip(theoretical_xlabels, lpc_theoretical_projected):
        nbr = Counter(l)
        for i in range(len(t)):
            lpc_freq[j].append((nbr.get(t[i], 0)) / length)  # returns the frequency of a key if exists otherwise zero!
        j += 1

    return lpc_freq


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
    # plt.show(block=False)
    # plt.pause(plot_time * 3)  # seconds
    # plt.close()


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
            lhc_freq = get_projected_freq(theoretical_xlabels, lhc_xlabels, groups)
            plot_side_by_side(theoretical_freq, theoretical_xlabels, lhc_freq, discrete, params, message, color='blue')
    else:
        if not SALib:
            message = f'Constant probabilistic parameters for all buildings with SALib-LHC'
        else:
            message = f'Constant probabilistic parameters for all buildings with Centered-LHC'
        lhc_freq = get_projected_freq(theoretical_xlabels, lhc_xlabels_all_buildings, groups)
        plot_side_by_side(theoretical_freq, theoretical_xlabels, lhc_freq, discrete, params, message, color='blue')

    print(f'\t+ plot_prior_distributions method is over!')


def plot_calibrated_parameters(data_frame, ranges, discrete, plot=True):
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
    message = f'Prior and posterior marginal distributions for calibrated parameters'
    lhc_freq = get_projected_freq(theoretical_xlabels, lhc_xlabels, groups)
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
    plt.annotate(str(round(y, 2))+"%", (np.argmin(error_to_plot), y),
                 xytext=(0, 10), textcoords="offset points", ha='center')

    plt.xlabel(f'Number of simulations for building {nbr}')
    plt.ylabel('Percentage of Error')
    fig = plt.gcf()
    perc = round((passed_size / len(error_to_plot)) * 100, 2)
    fig.canvas.set_window_title(f'{passed_size}/{len(error_to_plot)} ({perc}%) '
                                f'of tests with error lower than {alpha}%')
    plt.tight_layout()
    plt.show()
    # plt.show(block=False)
    # plt.pause(plot_time)
    # plt.close()


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
        color = 'g' if min_value <= alpha else 'r'
        plt.annotate((str(round(min_value, 2))+"%", id), (min_idx, min_value),
                     xytext=(0, 10), textcoords="offset points", ha='center', color=color)

    plt.xlabel(f'Number of all simulations for buildings: {[i for i in building_ids]}')
    plt.ylabel('Percentage of Error')
    fig = plt.gcf()
    perc = round((passed_size / len(all_errors_list)) * 100, 2)
    fig.canvas.set_window_title(f'{passed_size}/{len(all_errors_list)} ({perc}%) '
                                f'of tests with error lower than {alpha}%')
    plt.tight_layout()
    plt.show()
    # plt.show(block=False)
    # plt.pause(plot_time*2)
    # plt.close()


def get_simulation_runs(buildings, simulations):
    # prepare simulation results to compare - - -
    heat_el_result = {}  # Hot water consumption is not included!
    total_sim_result = {}

    for nbr in buildings:
        heat_el_result[nbr] = {}
        total_sim_result[nbr] = {}
        building_names = simulations[nbr].keys()

        for bn in building_names:
            heat_el_result[nbr][bn] = []
            tmp = 0
            for key, value in simulations[nbr][bn].items():
                if '_Electric' in key:
                    heat_el_result[nbr][bn].append(value)  # Kwh
                    tmp += value
                elif 'Ideal Loads Zone Total Heating' in key:
                    heat_el_result[nbr][bn].append(value)
                    tmp += value
            total_sim_result[nbr].update({bn: tmp})
    return total_sim_result


def compare_results(buildings, simulations, measurements, area,
                    alpha=5, per_building=False, all_buildings=True, plots=True):
    # prepare simulation results to compare - - -
    total_sim_result = get_simulation_runs(buildings, simulations)

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
            passed_cases_one_building(errors, nbr, alpha)

    if plots and all_buildings:
        passed_cases_all_buildings(errors, alpha)

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


def make_joint_distribution(buildings, acceptable, discrete, plot=True):
    print(f'\t- The start of Joint distribution method ...')
    os.chdir(res_path)
    accepted_ranges = {'Buildings': []}
    for b in buildings:
        to_open = list(acceptable[b].keys())
        for tmp in to_open:
            with open(tmp + '.pickle', 'rb') as handle:
                case = pickle.load(handle)
            case_BuildDB = case['BuildDB']

            for param in lb.VarName2Change:
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
        # g._legend.remove()
        plt.show()
        # plt.show(block=False)
        # plt.pause(plot_time * 6)
        # plt.close()

    # print(f'{df}')
    print(f'\t+ Joint distribution method is over!')
    return df


def plot_joint_distributions(data_frame, discrete):
    del data_frame['Buildings']

    sns.set_theme(style="white")
    cmap = sns.cubehelix_palette(light=1.5, as_cmap=True)

    header = list(data_frame.columns)
    graph = {}
    for i in range(len(header) - 1):
        graph[f'g{i + 1}'] = None

    i = 0
    for key in graph:
        graph[key] = sns.JointGrid(data=data_frame, x=header[0], y=header[i + 1], space=0)
        graph[key].plot_joint(sns.kdeplot, fill=True, clip=((lb.Bounds[0]), (lb.Bounds[i + 1])),
                              thresh=0, levels=discrete, cmap=cmap).plot_joint(sns.scatterplot)
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


# 'eigenvectors' -> cholesky=False
def make_random_from_correlated(data, var_names, nbr_samples, cholesky=True, plots=True):
    """
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
            "Pairwise plots of generated samples from covariance matrix of calibrated parameters")
        plt.show()

    tmp = np.array(y_transformed[0]).reshape((-1, 1))  # TRANSPOSE TO COLUMN FORMAT
    for i in range(len(y_transformed)-1):
        col_tmp = (y_transformed[i+1]).reshape(-1, 1)
        tmp = np.concatenate((tmp, col_tmp), axis=1)

    y_transformed = tmp
    print(f'\t+ Generating calibrated samples from correlated parameters is done in {round(time.time()-mrfc_time, 2)}s.')
    return y_transformed


def second_largest(numbers):
    # https://stackoverflow.com/questions/16225677/get-the-second-largest-number-in-a-list-in-linear-time
    count = 0
    m1 = m2 = float('-inf')
    for x in numbers:
        count += 1
        if x > m2:
            if x >= m1:
                m1, m2 = x, m1
            else:
                m2 = x
    return m2 if count >= 2 else None


def fit_best_distro(calib_params, nbr_samples, nbr_match, Danoe=True, plot=True):
    """
    The third parameter is used when plot option is set to 'True' for presenting the number of distributions
    This code is inspired by Sebastian Jose's answer in stackoverflow
    # https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python
    """
    read_time3 = time.time()
    print(f'\t- Finding the best distributions started ...')
    distributions = ['foldcauchy', 'cauchy', 'alpha', 'dweibull', 'genextreme', 'pearson3', 'dgamma']
    """
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
    """
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
        # predefined number of bins for each parameter in local_builder
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
                if second_largest(y_plot) * 10 < max(y_plot):
                    index_max = max(range(len(y_plot)), key=y_plot.__getitem__)
                    y_plot[index_max] = second_largest(y_plot) * 2

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
    for i in range(len(tmp)-1):
        col_tmp = (tmp[i+1][:nbr_samples]).reshape(-1, 1)
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

    print(f'- The start of annual calibration method ...')
    if not all_plots:
        print("** THE PLOT OPTION IS OFF, CHANGE TO 'TRUE' TO SEE THE RESULTS! **")
    # * ALGORITHM STEP1: PARAMETER DEFINITION * * * * * * * * * * /
    _, a_temp, total_epc, _, _, params_cat, params_build = read_epc_values(lb.VarName2Change, 0)
    if all_plots:
        plot_prior_distributions(params, params_ranges, buildings, discrete,
                                 per_building=False, SALib=lb.SAMPLE_TYPE, LHC=params_cat)

    # * ALGORITHM STEP2: PARAMETRIC SIMULATION * * * * * * * * * * /
    print(f"\t-> {nbr_sim_uncalib} random simulations out of {discrete ** (len(params))} possible combinations!")
    sim_data, model_area = read_simulation_files()

    # * ALGORITHM STEP3: ERROR QUALIFICATION (α) * * * * * * * * * * /
    total_sim_results, _, errors, acceptable = \
        compare_results(buildings, sim_data, total_epc, model_area, alpha=alpha,
                        per_building=True, all_buildings=True, plots=all_plots)

    # * ALGORITHM STEP4: TEST OF ASSUMPTIONS (β) * * * * * * * * * * /
    unexplained_buildings = []
    ratio = 0
    for b in buildings:
        if len(acceptable[b]) >= 1:
            ratio += 1
        else:
            unexplained_buildings.append(b)

    percentage = (ratio / len(buildings)) * 100
    if percentage > beta:
        print(f"\t-> β qualification satisfied with {percentage}%.")
    else:
        print(f"-> ATTENTION: only {percentage}% simulations of buildings matched measurements!\n"
              f"-> Model revision, choice of θ parameters or more number of simulations is needed!\n"
              f"-> List of unexplained buildings: {unexplained_buildings}")
        return 1

    # * ALGORITHM STEP5: DISTRIBUTION GENERATION * * * * * * * * * * /
    calib_params = make_joint_distribution(buildings, acceptable, discrete, plot=all_plots)

    # Plot below provides more insight to possible correlations of parameters /
    # if all_plots: # FIXME: AN EXTRA PLOT, CORRELATION OF PARAMETERS, UNCOMMENT TO SEE.
    #    plot_joint_distributions(calib_params, discrete)

    calib_frequencies = plot_calibrated_parameters(calib_params, lb.Bounds, discrete, plot=all_plots)

    # * ALGORITHM STEP6: RANDOM SAMPLED SIMULATIONS * * * * * * * * * * /
    # Save and send the result back to local_builder for simulations with θ'
    smpl = final_samples  # The number of final samples based on calibrated parameters

    if approach == 1:
        theta_prime = make_random_from_correlated(calib_params, lb.VarName2Change, smpl,
                                                   cholesky=True, plots=all_plots)
        np.savetxt("theta_prime.csv", theta_prime, delimiter=",")

    elif approach == 2:
        best_distros = fit_best_distro(calib_params, lb.sample_nbr, 3, Danoe=False, plot=all_plots)
        theta_prime = make_random_from_continuous(best_distros, lb.Bounds, smpl, plot=all_plots)
        np.savetxt("theta_prime.csv", theta_prime, delimiter=",")

    print(f'+ Calibration method is over!')
    return 0


# CALIBRATION RUN * * * * * * * *
if __name__ == '__main__':
    if lb.RUN_UNSEEN_BUILDINGS_WITH_CALIBRATED_PARAMETERS:
        print(f'* Simulation results for buildings: {lb.BuildNum} with calibrated parameters *')
        _, a_temp, total_epc, _, _, params_cat, params_build = read_epc_values(lb.VarName2Change, 1)
        sim_data, model_area = read_simulation_files()
        total_sim_results, _, errors, acceptable = \
            compare_results(lb.BuildNum, sim_data, total_epc, model_area, alpha=2,
                            per_building=True, all_buildings=True, plots=True)

        # TODO: BELOW ARE DUMMY VALUES, FIX IT!
        n = 1000
        u1 = 5
        u2 = 5
        series1 = u1 + np.random.randn(n)
        series2 = u2 + np.random.randn(n)
        plot_metered_vs_simulated_energies(series1, series2, bins=20)
        print(f'** Illustration of the results by calibrated parameters is over *')
    else:
        # plot_time = 5  # plots terminate in a plot_time by a multiplayer!
        calibrate_uncertain_params(lb.VarName2Change, lb.Bounds, lb.NbRuns, lb.BuildNum, alpha=5,
                    beta=90, final_samples=100, discrete=lb.sample_nbr, all_plots=True, approach=1)

        # Prior presentation of real samples for selected parameters
        if lb.SAMPLE_TYPE:
            combinations, middle_points = lb.all_combinations(lb.Bounds, lb.VarName2Change, lb.sample_nbr)
            plot_combinations(lb.VarName2Change, lb.Bounds, combinations, middle_points, lb.sample_nbr)

        print(f"* Execution time:{round((time.time() - start_time), 2)}s /"
              f" {round(((time.time() - start_time) / 60), 2)}min!")
