# **********************************
# The script in this file belongs to
#      Dr. Xavier Faure @ KTH
#          <xavierf@kth.se>
# **********************************
import time
import local_utility as lu
from local_setup import *
import local_setup as ls

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


class LocalBuilder:
    def __init__(self):
        print("*** AN INSTANTIATION OF LocalBuilder class ***")

    def appendBuildCase(self, StudiedCase, epluspath, nbcase, Buildingsfile, Shadingsfile, MainPath):
        StudiedCase.addBuilding('Building' + str(nbcase), Buildingsfile, Shadingsfile, nbcase, MainPath, epluspath)
        idf = StudiedCase.building[-1]['BuildIDF']
        building = StudiedCase.building[-1]['BuildData']
        return idf, building

    # Simulation Level -------------
    def setSimLevel(self, idf, building):
        Sim_param.Location_and_weather(idf, building)
        Sim_param.setSimparam(idf)

    # Building Level -------------
    def setBuildingLevel(self, idf, building):
        # this is the function that requires the most time
        GeomScripts.createBuilding(idf, building, perim=False)

    # Building to Envelope Level -------------
    def setEnvelopeLevel(self, idf, building):
        # the other geometric element are thus here (within the building level)
        GeomScripts.createRapidGeomElem(idf, building)

    # Zone level -------------
    def setZoneLevel(self, idf, building, MainPath):
        # control command related equipment, loads and leaks for each zones
        Load_and_occupancy.CreateZoneLoadAndCtrl(idf, building, MainPath)

    # Output level -------------
    def setOutputLevel(self, idf, MainPath):
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

    def readPathfile(self):
        keyPath = {'epluspath': '', 'Buildingsfile': '', 'Shadingsfile': ''}
        with open('local_pathways.txt', 'r') as PathFile:
            Paths = PathFile.readlines()
            for line in Paths:
                for key in keyPath:
                    if key in line:
                        keyPath[key] = os.path.normcase(line[line.find(':') + 1:-1])

        return keyPath

    def LaunchProcess(self, bldidx, keyPath, nbcase, VarName2Change=[], Bounds=[], nbruns=1, CPUusage=1, sample_nbr=3,
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

        # Sampling process if you define int as function's arguments
        # It is currently using the latin hyper cube methods for the sampling generation (latin.sample)
        Param = [1]
        if len(VarName2Change) > 0:
            problem = {}
            problem['names'] = VarName2Change
            problem['bounds'] = Bounds
            problem['num_vars'] = len(VarName2Change)
            # problem = read_param_file(MainPath+'\\liste_param.txt')

            if ls.CALIBRATE_WITH_CALIBRATED_PARAMETERS:
                Param = lu.read_calibrated_params(os.getcwd(), CaseName, recursive=RECURSIVE_CALIBRATION)
                print(f'Simulations with calibrated parameters ...')
            else:
                Param = latin.sample(problem, nbruns)

        Res = {}
        # this will be the final list of studied cases : list of objects stored in a dict . idf key for idf object and building key for building database object
        # even though this approache might be not finally needed as I didnt manage to save full object in a pickle and reload it for launching.
        # see LaunchSim.runcase()
        # Nevertheless this organization still enable to order things !
        StudiedCase = BuildingList()
        # lets build the two main object we'll be playing with in the following'
        idf_ref, building_ref = self.appendBuildCase(StudiedCase, epluspath, nbcase, Buildingsfile, Shadingsfile,
                                                     MainPath)

        if building_ref.height == 0:
            print('This Building does not have any height, process abort for this one')
            os.chdir(MainPath)
            return MainPath, epluspath
        # change on the building __init__ class in the simulation level should be done here
        self.setSimLevel(idf_ref, building_ref)
        # change on the building __init__ class in the building level should be done here
        self.setBuildingLevel(idf_ref, building_ref)

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
            self.setEnvelopeLevel(idf, building)

            # just uncomment the line below if some 3D view of the building is wanted. The figure's window will have to be manually closed for the process to continue
            # idf.view_model(test=False)

            # change on the building __init__ class in the zone level should be done here
            self.setZoneLevel(idf, building, MainPath)

            self.setOutputLevel(idf, MainPath)

            # saving files and objects
            idf.saveas('Building_' + str(nbcase) + 'v' + str(i) + '.idf')
            with open('Building_' + str(nbcase) + 'v' + str(i) + '.pickle', 'wb') as handle:
                pickle.dump(Case, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('Input IDF file ', i + 1, '/', len(Param), ' is done')
        # RunProcess(MainPath, epluspath, CPUusage)

        # lets get back to the Main Folder we were at the very beginning
        os.chdir(MainPath)
        return MainPath, epluspath, Param

    def run(self, case_name):
        start_time = time.time()
        ###################################################################################################################
        ########        MAIN INPUT PART     ###############################################################################
        ###################################################################################################################
        # The Modeler have to fill in the following parameter to define its choices

        # CaseName = 'String'                   #name of the current study (the output folder will be renamed using this entry)
        # BuildNum = [1,2,3,4]                  #list of numbers : number of the buildings to be simulated (order respecting the
        #                                       geojson file)
        # VarName2Change = ['String','String']  #list of strings: Variable names (same as Class Building attribute, if different
        #                                       see LaunchProcess 'for' loop for examples)
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
        keyPath = self.readPathfile()
        for idx, nbBuild in enumerate(BuildNum):
            print('Building ' + str(nbBuild) + ' is starting')
            MainPath, epluspath, sensitivity_input = self.LaunchProcess(idx, keyPath, nbBuild, VarName2Change, Bounds,
                                                                        NbRuns,
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
                          os.path.join(os.getcwd(), case_name + '_Build_' + str(nbBuild)))
        if not SepThreads:
            file2run = LaunchSim.initiateprocess(MainPath)
            nbcpu = max(mp.cpu_count() * CPUusage, 1)
            pool = mp.Pool(processes=int(nbcpu))  # let us allow 80% of CPU usage
            for i in range(len(file2run)):
                pool.apply_async(LaunchSim.runcase, args=(file2run[i], MainPath, epluspath))
            pool.close()
            pool.join()
            os.rename(os.path.join(os.getcwd(), 'RunningFolder'), os.path.join(os.getcwd(), case_name))
        # lets suppress the path we needed for geomeppy
        # sys.path.remove(self.path2addgeom) # TODO: ASK XAVIER!

        print(f"Total execution time: {round((time.time() - start_time), 2)} s / "
              f"{round(((time.time() - start_time) / 60), 2)} min "
              f"for {NbRuns} simulations per building!")  # * len(VarName2Change) * len(BuildNum)


if SIMULATE_TO_CALIBRATE:
    one_time_calibration = LocalBuilder()
    one_time_calibration.run(CaseName)
