import os, time, pickle, copy
import local_builder as lb
import numpy as np, seaborn as sns, pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import pearsonr

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
    alternative to 0 -> based on generated input file names
    alternative to 1 -> based on existing simulation file names
    This function also checks the existing number of files against the number of simulations asked by user!
    """
    # check the number of simulations in the result folder and produce their names
    all_files = os.listdir(path)
    if NbRuns * len(BuildNum) != len(all_files) / 2:
        print(f"\n-> Something is WRONG, number of exiting files does not equal to 'BuildNum'!")
        print(f'-> Result pickle files:{len(all_files) / 2} != Simulations asked:{NbRuns * len(BuildNum)}')
        raise ValueError("Missing simulation files!")

    lst = {}
    if alternative:
        # find simulation file names to extract energy results
        # method 1
        for file in all_files:
            if file.endswith('.pickle'):
                for cn in BuildNum:
                    if str(cn) in file:  # if file.find(cn):
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
    input_files = get_file_names(lb.BuildNum, lb.NbRuns, res_path, 0)
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


def plot_prior_distributions(params, ranges, discrete, **kwargs):
    print(f'\t- The start of plot_prior_distributions method ...')
    groups = len(params)
    theoretical_freq = [[1 / discrete] * discrete] * groups
    lhc = kwargs.get('LHC')  # Latin Hyper Cube Frequencies
    b_nbr = kwargs.get('Case')  # Building number
    message = kwargs.get('Message')

    theoretical_xlabels = []
    for pr in ranges:
        if pr[0] < 0.01:
            theoretical_xlabels.append(list(np.round(np.linspace(pr[0], pr[1], discrete), 3)))
        else:
            theoretical_xlabels.append(list(np.round(np.linspace(pr[0], pr[1], discrete), 2)))

    # - - - - - - - - - - - - - - - - - - - -
    # The block below fits random numbers from LHC to theoretical distribution range
    lpc_xlabels = []
    lpc_theoretical_projected = []
    for param in params:
        lpc_xlabels.append(lhc[b_nbr][param])
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
    # - - - - - - - - - - - - - - - - - - - -

    x = np.arange(discrete)  # position of groups, using X to align the bars side by side

    width = 0.75
    fig, axs = plt.subplots(1, groups, figsize=(7, 4.5), sharey=True)

    for n, ax in enumerate(axs):
        ax.bar(x, theoretical_freq[n], color='gray', alpha=0.7,
               width=width, align='center', edgecolor='white')
        ax.bar(x, lpc_freq[n], color='blue', alpha=0.5,
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
    print(f'\t+ plot_prior_distributions method is over!')


def passed_cases_one_building(error_dict, nbr, alpha):
    # plot error values
    error_to_plot = np.fromiter(error_dict[nbr].values(), dtype=float)
    passed_size = len(error_to_plot[error_to_plot < alpha])
    sim_size = range(len(error_to_plot))

    plt.figure(figsize=(7, 4), dpi=80)
    error_colors = np.where(error_to_plot < alpha, 'g', 'k')
    plt.scatter(sim_size, error_to_plot, c=error_colors)
    plt.axhline(y=alpha, color='r', linestyle='--')

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
    for id in building_ids:
        tmp = list(np.fromiter(error_dict[id].values(), dtype=float))
        all_errors_list += tmp
    all_errors_list = np.array(all_errors_list)

    passed_size = len(all_errors_list[all_errors_list < alpha])
    sim_size = range(len(all_errors_list))

    plt.figure(figsize=(7, 4), dpi=80)
    error_colors = np.where(all_errors_list < alpha, 'g', 'k')
    plt.scatter(sim_size, all_errors_list, c=error_colors)
    plt.axhline(y=alpha, color='r', linestyle='--')

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


def compare_results(buildings, simulations, measurements, area, alpha, plots):
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

        if plots:
            # comment the line below out in case of many buildings!
            # passed_cases_one_building(errors, nbr, alpha)
            pass
    if plots:
        passed_cases_all_buildings(errors, alpha)

    # KEEP acceptable buildings with respect to alpha!
    acceptable = {}
    for nbr in buildings:
        temp = dict((k1, v1) for k1, v1 in errors[nbr].items() if v1 <= alpha)
        acceptable.update({nbr: temp})

    return total_sim_result, reference, errors, acceptable


def get_correlation(x1, y1, **kws):
    (r, p) = pearsonr(x1, y1)
    ax = plt.gca()
    ax.annotate("r={:.2f}".format(r), xy=(.1, .9), xycoords=ax.transAxes)
    ax.annotate("p={:.3f}".format(p), xy=(.1, .8), xycoords=ax.transAxes)


def make_joint_distribution(buildings, acceptable, plots):
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

    if plots:
        # https://seaborn.pydata.org/tutorial/color_palettes.html
        df_tmp = df.copy()
        df_tmp.loc[:, 'Buildings'] = 'Explained'
        g = sns.pairplot(df_tmp, hue="Buildings", palette="Set2", diag_kind="kde", height=1.5)
        g.map_diag(sns.histplot, hue=None, color=".3")
        # g.map_offdiag(sns.regplot)
        g.map_upper(sns.regplot)
        g.map_lower(sns.kdeplot, levels=5, color="k", shade=True)
        fig = plt.gcf()
        fig.canvas.set_window_title(f'Possible correlations among {df_tmp.shape[0]}'
                                    f' validated combination of parameters')
        g.fig.set_size_inches(10, 7)
        g.map(get_correlation)
        # g._legend.remove()
        plt.show()
        # plt.show(block=False)
        # plt.pause(plot_time * 6)
        # plt.close()
    else:
        print(f'{df}')

    print(f'\t+ Joint distribution method is over!')
    return df


def plot_joint_distributions(data_frame):
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


def calibrate_uncertain_params(params, params_ranges, nbr_sim, buildings, alpha=5, beta=85, discrete=5, plots=True):
    """
    This function is based on Carlos Cerezo, et al's method published in 2017
    This method needs:
    'params' -> uncertain parameter names
    'params_ranges' -> uncertain parameter ranges
    'alpha' value -> the upper acceptable limit for simulation and measurement difference for each building
    'beta' value -> minimum percentage of acceptable simulations which satisfies 'alpha' condition
    'discrete' -> as the number of division for continuous parameter ranges
    """

    print(f'- The start of annual calibration method ...')
    # * ALGORITHM STEP1: PARAMETER DEFINITION * * * * * * * * * * /
    _, a_temp, total_epc, _, _, params_cat, params_build = read_epc_values(lb.VarName2Change, 0)

    if plots:
        for b in buildings:
            message = f'Constant probabilistic parameters vs LHC in building {b}'
            plot_prior_distributions(params, params_ranges, discrete, LHC=params_cat, Case=b, Message=message)

    # * ALGORITHM STEP2: PARAMETRIC SIMULATION * * * * * * * * * * /
    print(f"-> {nbr_sim} random simulations out of {discrete ** (len(params))} possible combinations!")
    sim_data, model_area = read_simulation_files()

    # * ALGORITHM STEP3: ERROR QUALIFICATION (α) * * * * * * * * * * /
    total_sim_results, _, errors, acceptable = compare_results(buildings, sim_data, total_epc, model_area, alpha, plots)

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
        print(f"-> β qualification satisfied with {percentage}%.")
    else:
        print(f"-> ATTENTION: only {percentage}% simulations of buildings matched measurements!\n"
              f"-> Model revision, choice of θ parameters or more number of simulations is needed!\n"
              f"-> List of unexplained buildings: {unexplained_buildings}")
        return 1

    # * ALGORITHM STEP5: DISTRIBUTION GENERATION * * * * * * * * * * /
    joint_dist = make_joint_distribution(buildings, acceptable, plots)
    plot_joint_distributions(joint_dist)

    # * ALGORITHM STEP6: RANDOM SAMPLED SIMULATIONS * * * * * * * * * * /

    # TODO: Send the result back to local_builder for simulations with θ'

    print(f'+ Calibration method is over!')
    return 0


# CALIBRATION RUN * * * * * * * *
if __name__ == '__main__':
    # plot_time = 5  # plots terminate in a plot_time by a multiplayer!
    calibrate_uncertain_params(lb.VarName2Change, lb.Bounds, lb.NbRuns, lb.BuildNum,
                               alpha=15, beta=90, discrete=5, plots=True)

    print(f"* Execution time:{round((time.time() - start_time), 2)}s /"
          f" {round(((time.time() - start_time) / 60), 2)}min!")
