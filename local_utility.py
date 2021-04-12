import os, time, pickle, copy
import local_builder as lb
import numpy as np
from collections import Counter


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

    if os.path.isfile(path + "/best_combinations.csv"):
        os.remove(path + "/best_combinations.csv")
        print("\t-> Previous 'best_combinations.csv' file is removed.")

    # check the number of simulations in the result folder and produce their names
    all_files = os.listdir(path)
    if NbRuns * len(BuildNum) != len(all_files) / 2:
        print(f"\t-> Something is WRONG, number of exiting files does not equal to 'BuildNum'!")
        print(f'\t-> Parameters in local_builder should match with existing number of files in ForTest folder!')
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


def read_epc_values(params_list, surface_type, path):
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
    input_files = get_file_names(lb.BuildNum, lb.NbRuns, path, 1)
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


def read_simulation_files(path):
    read_time2 = time.time()
    print(f'\t- Reading of result files with pickle format started ...')
    result_files = get_file_names(lb.BuildNum, lb.NbRuns, path, 1)

    os.chdir(path)  # change the working directory to result files!

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


def second_largest(numbers):
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

