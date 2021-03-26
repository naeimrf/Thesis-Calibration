# **********************************
# The script in this file belongs to
#      Dr. Xavier Faure @ KTH
#          <xavierf@kth.se>
# **********************************
import os
import sys
import time

start_time = time.time()
# add the required path
path2addgeom = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'geomeppy')
sys.path.append(path2addgeom)
# add needed packages
import pygeoj
import pickle
import copy
from SALib.sample import latin

# add scripts from the project as well
sys.path.append("..")
import CoreFiles.GeomScripts as GeomScripts
import CoreFiles.Set_Outputs as Set_Outputs
import CoreFiles.Sim_param as Sim_param
import CoreFiles.Load_and_occupancy as Load_and_occupancy
import CoreFiles.LaunchSim as LaunchSim
from DataBase.DB_Building import BuildingList
import multiprocessing as mp


def appendBuildCase(StudiedCase, epluspath, nbcase, Buildingsfile, Shadingsfile, MainPath):
    StudiedCase.addBuilding('Building' + str(nbcase), Buildingsfile, Shadingsfile, nbcase, MainPath, epluspath)
    idf = StudiedCase.building[-1]['BuildIDF']
    building = StudiedCase.building[-1]['BuildData']
    return idf, building


# Simulation Level -------------
def setSimLevel(idf, building):
    Sim_param.Location_and_weather(idf, building)
    Sim_param.setSimparam(idf)


# Building Level -------------
def setBuildingLevel(idf, building):
    # this is the function that requires the most time
    GeomScripts.createBuilding(idf, building, perim=False)


# Building to Envelope Level -------------
def setEnvelopeLevel(idf, building):
    # the other geometric element are thus here (within the building level)
    GeomScripts.createRapidGeomElem(idf, building)


# Zone level -------------
def setZoneLevel(idf, building, MainPath):
    # control command related equipment, loads and leaks for each zones
    Load_and_occupancy.CreateZoneLoadAndCtrl(idf, building, MainPath)


# Output level -------------
def setOutputLevel(idf, MainPath):
    # outputs definitions
    Set_Outputs.AddOutputs(idf, MainPath)


# def RunProcess(MainPath,epluspath,CPUusage):
#     file2run = LaunchSim.initiateprocess(MainPath)
#     MultiProcInputs={'file2run' : file2run,
#                      'MainPath' : MainPath,
#                      'CPUmax' : CPUusage,
#                      'epluspath' : epluspath}
#     #we need to picke dump the input in order to have the protection of the if __name__ == '__main__' : in LaunchSim file
#     #so the argument are saved into a pickle and reloaded in the main (see if __name__ == '__main__' in LaunchSim file)
#     with open(os.path.join(MainPath, 'MultiProcInputs.pickle'), 'wb') as handle:
#         pickle.dump(MultiProcInputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     LaunchSim.RunMultiProc(file2run, MainPath, False, CPUusage,epluspath)

def readPathfile():
    keyPath = {'epluspath': '', 'Buildingsfile': '', 'Shadingsfile': ''}
    with open('local_pathways.txt', 'r') as PathFile:
        Paths = PathFile.readlines()
        for line in Paths:
            for key in keyPath:
                if key in line:
                    keyPath[key] = os.path.normcase(line[line.find(':') + 1:-1])

    return keyPath


#####################################################
# Added by Naeim
from skopt.space import Space
from skopt.sampler import Lhs
import pandas as pd
from itertools import product
import numpy as np
def all_combinations(bounds, var_names, samples):
    """
    This function takes probabilistic variable bounds, number of discritization and variable names
    and returns a list of lists for all combinations.
    """
    #  converting list of lists to list of tuples suitable for Space function in skopt!
    tmp = []
    for bound in bounds:
        tmp.append([float(b) for b in bound])
    bounds = list(map(tuple, tmp))
    space = Space(bounds)

    # LHC with centered option to get the middle points
    lhs = Lhs(lhs_type="centered", criterion=None)
    centered = lhs.generate(space.dimensions, samples)

    # centered is a list of lists where each list has one value for each probabilistic variable
    # convert to dataframe to split columns into separated variables
    df = pd.DataFrame(centered, columns=var_names)
    inverted_list = []
    params = df.columns.values.tolist()
    for param in params:
        inverted_list.append(df[param].tolist())

    # producing all combinations by product from itertools
    # the result of product function is a list of tuples
    combinations = list(product(*inverted_list))
    tmp = []
    for comb in combinations:
        tmp.append(list(comb))
    combinations = tmp

    return combinations, inverted_list


#####################################################


def LaunchProcess(bldidx, keyPath, nbcase, VarName2Change=[], Bounds=[], nbruns=1, CPUusage=1, sample_nbr=3,
                  SepThreads=True):
    # this main is written for validation of the global workflow. and as an example for other simulation
    # the cases are build in a for loop and then all cases are launched in a multiprocess mode, the maximum %of cpu is given as input
    MainPath = os.getcwd()
    epluspath = keyPath['epluspath']
    Buildingsfile = pygeoj.load(keyPath['Buildingsfile'])
    Shadingsfile = pygeoj.load(keyPath['Shadingsfile'])

    SimDir = os.path.join(os.getcwd(), 'RunningFolder')
    if not os.path.exists(SimDir):
        os.mkdir(SimDir)
    elif SepThreads or bldidx == 0:
        for i in os.listdir(SimDir):
            if os.path.isdir(os.path.join(SimDir, i)):
                for j in os.listdir(os.path.join(SimDir, i)):
                    os.remove(os.path.join(os.path.join(SimDir, i), j))
                os.rmdir(os.path.join(SimDir, i))
            else:
                os.remove(os.path.join(SimDir, i))
    os.chdir(SimDir)

    # Sampling process if someis define int eh function's arguments
    # It is currently using the latin hyper cube methods for the sampling generation (latin.sample)
    Param = [1]
    if len(VarName2Change) > 0:
        problem = {}
        problem['names'] = VarName2Change
        problem['bounds'] = Bounds
        problem['num_vars'] = len(VarName2Change)
        # problem = read_param_file(MainPath+'\\liste_param.txt')
        if SAMPLE_TYPE:
            Param, _ = all_combinations(Bounds, VarName2Change, sample_nbr)
            Param = np.array([np.array(x) for x in Param])
            print(f'Selected sample method -> sikit optimization LHC-Centered')
        else:
            Param = latin.sample(problem, nbruns)
            print(f'Selected sample method -> SALib')

    Res = {}
    # this will be the final list of studied cases : list of objects stored in a dict . idf key for idf object and building key for building database object
    # even though this approache might be not finally needed as I didnt manage to save full object in a pickle and reload it for launching.
    # see LaunchSim.runcase()
    # Nevertheless this organization still enable to order things !
    StudiedCase = BuildingList()
    # lets build the two main object we'll be playing with in the following'
    idf_ref, building_ref = appendBuildCase(StudiedCase, epluspath, nbcase, Buildingsfile, Shadingsfile, MainPath)

    if building_ref.height == 0:
        print('This Building does not have any height, process abort for this one')
        os.chdir(MainPath)
        return MainPath, epluspath
    # change on the building __init__ class in the simulation level should be done here
    setSimLevel(idf_ref, building_ref)
    # change on the building __init__ class in the building level should be done here
    setBuildingLevel(idf_ref, building_ref)

    # now lets build as many cases as there are value in the sampling done earlier
    for i, val in enumerate(Param):
        # we need to copy the reference object because there is no need to set the simulation level nor the building level
        # (except if some wanted and thus the above function will have to be in the for loop process
        idf = copy.deepcopy(idf_ref)
        building = copy.deepcopy(building_ref)
        idf.idfname = 'Building_' + str(nbcase) + 'v' + str(i)
        Case = {}
        Case['BuildIDF'] = idf
        Case['BuildData'] = building

        # # example of modification with half of the runs with external insulation and half of the runs with internal insulation
        # if i < round(nbruns / 2):
        #     building.ExternalInsulation = True
        # else:
        #     building.ExternalInsulation = False

        # now lets go along the VarName2Change list and change the building object attributes
        # if these are embedded into several layer dictionary then there is a need to check and change accordingly the c
        # orrect element here are examples for InternalMass impact using 'InternalMass' keyword in the VarName2Change list
        # to play with the 'WeightperZoneArea' parameter and for ExternalMass impact using 'ExtMass' keyword in the
        # VarName2Change list to play with the 'Thickness' of the wall inertia layer
        for varnum, var in enumerate(VarName2Change):
            if 'InternalMass' in var:
                intmass = building.InternalMass
                intmass['HeatedZoneIntMass']['WeightperZoneArea'] = Param[i, varnum]
                setattr(building, var, intmass)
            elif 'ExtMass' in var:
                exttmass = building.Materials
                exttmass['Wall Inertia']['Thickness'] = round(Param[i, varnum] * 1000) / 1000
                setattr(building, var, exttmass)
            else:
                setattr(building, var, Param[i, varnum])
            # for all other cases with simple float just change the attribute's value directly
            # here is an other example for changing the distince underwhich the surrounding building are considered for shading aspects
            # as 'MaxShadingDist' is an input for the Class building method getshade, the method shall be called again after modifying this value (see getshade methods)
            if 'MaxShadingDist' in var:
                building.shades = building.getshade(Buildingsfile[nbcase], Shadingsfile, Buildingsfile)

        ##############################################################
        ##After having made the changes we wanted in the building object, we can continue the construction of the idf (input file for EnergyPLus)

        # change on the building __init__ class in the envelope level should be done here
        setEnvelopeLevel(idf, building)

        # just uncomment the line below if some 3D view of the building is wanted. The figure's window will have to be manually closed for the process to continue
        # idf.view_model(test=False)

        # change on the building __init__ class in the zone level should be done here
        setZoneLevel(idf, building, MainPath)

        setOutputLevel(idf, MainPath)

        # saving files and objects
        idf.saveas('Building_' + str(nbcase) + 'v' + str(i) + '.idf')
        with open('Building_' + str(nbcase) + 'v' + str(i) + '.pickle', 'wb') as handle:
            pickle.dump(Case, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Input IDF file ', i + 1, '/', len(Param), ' is done')
    # RunProcess(MainPath, epluspath, CPUusage)

    # lets get back to the Main Folder we were at the very beginning
    os.chdir(MainPath)
    return MainPath, epluspath, Param


CaseName = 'ForTest'  # a folder_name to save input and outfile in form of pickles!
BuildNum = [10]
# check parameters names under DB_Building!
VarName2Change = ['EnvLeak', 'setTempLoL', 'wwr', 'BasementAirLeak']
Bounds = [[0.4, 2], [20, 23], [0.15, 0.35], [0.05, 2]]
NbRuns = 81

# SALib or SKOPT LHC-Centered
SAMPLE_TYPE = False  # False -> SALib, True -> SKOPT LHC-Centered
sample_nbr = 3  # number of middle point samples in each parameter range!
if SAMPLE_TYPE:
    NbRuns = sample_nbr ** len(VarName2Change)

if __name__ == '__main__':
    ###################################################################################################################
    ########        MAIN INPUT PART     ###############################################################################
    ###################################################################################################################
    # The Modeler have to fill in the following parameter to define its choices

    # CaseName = 'String'                   #name of the current study (the ouput folder will be renamed using this entry)
    # BuildNum = [1,2,3,4]                  #list of numbers : number of the buildings to be simulated (order respecting the
    #                                       geojsonfile)
    # VarName2Change = ['String','String']  #list of strings: Variable names (same as Class Building attribute, if different
    #                                       see LaunchProcess 'for' loopfor examples)
    # Bounds = [[x1,y1],[x2,y2]]            #list of 2 values list :bounds in which the above variable will be allowed to change
    # NbRuns = 1000                         #number of run to launch for each building (all VarName2Change will have automatic
    #                                       allocated value (see sampling in LaunchProcess)
    # CPUusage = 0.7                        #factor of possible use of total CPU for multiprocessing. If only one core is available,
    #                                       this value should be 1
    # SepThreads = False / True             #True = multiprocessing will be run for each building and outputs will have specific
    #                                       folders (CaseName string + number of the building. False = all input files for all
    #                                       building will be generated first, all results will be saved in one single folder

    CPUusage = 1
    SepThreads = False

    ##############################################################
    ########     LAUNCHING MULTIPROCESS PROCESS PART     #########
    ##############################################################
    keyPath = readPathfile()
    for idx, nbBuild in enumerate(BuildNum):
        print('Building ' + str(nbBuild) + ' is starting')
        MainPath, epluspath, sensitivity_input = LaunchProcess(idx, keyPath, nbBuild, VarName2Change, Bounds, NbRuns,
                                                               CPUusage, sample_nbr, SepThreads)
        if SepThreads:
            file2run = LaunchSim.initiateprocess(MainPath)
            nbcpu = max(mp.cpu_count() * CPUusage, 1)
            pool = mp.Pool(processes=int(nbcpu))  # let us allow 80% of CPU usage
            for i in range(len(file2run)):
                pool.apply_async(LaunchSim.runcase, args=(file2run[i], MainPath, epluspath))
            pool.close()
            pool.join()
            os.rename(os.path.join(os.getcwd(), 'RunningFolder'),
                      os.path.join(os.getcwd(), CaseName + '_Build_' + str(nbBuild)))
    if not SepThreads:
        file2run = LaunchSim.initiateprocess(MainPath)
        nbcpu = max(mp.cpu_count() * CPUusage, 1)
        pool = mp.Pool(processes=int(nbcpu))  # let us allow 80% of CPU usage
        for i in range(len(file2run)):
            pool.apply_async(LaunchSim.runcase, args=(file2run[i], MainPath, epluspath))
        pool.close()
        pool.join()
        os.rename(os.path.join(os.getcwd(), 'RunningFolder'), os.path.join(os.getcwd(), CaseName))
    # lets suppress the path we needed for geomeppy
    sys.path.remove(path2addgeom)

    print(f"Total execution time: {round((time.time() - start_time), 2)} s / "
          f"{(time.time() - start_time) / 60} min "
          f"for {NbRuns} simulations!")  # * len(VarName2Change) * len(BuildNum)
