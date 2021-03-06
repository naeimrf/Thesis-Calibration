import sys, os
"""
This script is a central place to hold initial parameters for all local scripts!
"""
# add the required path
path2addgeom = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'geomeppy')
sys.path.append(path2addgeom)

# RUNNING SENSITIVITY ANALYSIS IS COMPLETELY INDEPENDENT OF BOOLEANS IN OPTION 1 AND 2, ONLY SET:
#   -> CaseName, BuildNum, VarName2Change, Bounds AND NbRuns BASED ON YOUR ALREADY SIMULATED FILES!

# OPTION 1: ALWAYS KEEP RECURSIVE_CALIBRATION = False <-
# 1. FIRST RUN
#   -> SIMULATE_TO_CALIBRATE = True AND CALIBRATE_WITH_CALIBRATED_PARAMETERS = False THEN RUN local_builder.py
# - - - - - - - - - - - - - - -
# 2. TO CALIBRATE PARAMETERS
#   -> SIMULATE_TO_CALIBRATE = False AND CALIBRATE_WITH_CALIBRATED_PARAMETERS = False THEN RUN local_calib.py
# - - - - - - - - - - - - - - -
# 3. TO SIMULATE WITH CALIBRATED PARAMETERS
#   -> PUT CALIBRATED CSV FILE IN CSV FOLDER IN ModelerFolder FIRST!
#   -> SIMULATE_TO_CALIBRATE = True AND CALIBRATE_WITH_CALIBRATED_PARAMETERS = True THEN RUN local_builder.py
# - - - - - - - - - - - - - - -
# 4. TO SEE THE RESULT OF CALIBRATION
#   -> SIMULATE_TO_CALIBRATE = False AND CALIBRATE_WITH_CALIBRATED_PARAMETERS = True THEN RUN local_calib.py

SIMULATE_TO_CALIBRATE = False
CALIBRATE_WITH_CALIBRATED_PARAMETERS = False

# OPTION 2: SET THE BOOLEAN BELOW TO 'TRUE' TO RUN 'local_builder' RECURSIVELY, (EXPERIMENTAL CODE)
# UNTIL ALL BUILDINGS CAN SATISFY BETA VALUE IN CALIBRATION ALGORITHM!
# THIS OPTION IS SUITABLE FOR SEVERAL-TIME CALIBRATIONS BUT WITH LOW NUMBER OF SIMULATIONS!
RECURSIVE_CALIBRATION = False


CaseName = 'ForTest_9_10_126st'  # a folder_name to save input and outfile in form of pickles!
BuildNum = [9, 10]  # building numbers (names) to be simulated  #  [i for i in range(7, 28)]
# Before each run, pay attention to Outputs.txt (lines: 110, 353, 359)
VarName2Change = ['EnvLeak', 'wwr', 'IntLoadMultiplier', 'AreaBasedFlowRate', 'setTempLoL']
Bounds = [[0.4, 2], [0.2, 0.4], [0.5, 2], [0.3, 0.6], [19, 22]]

# Morris analysis needs 'NbRuns' to be a multiplier of 'number of parameters'
# That is if you have 5 parameters you need (5+1) * k number simulations. Ex: 17x6=102
NbRuns = 126

# ** EXPERIMENTAL CODE **
# SALib or SKOPT LHC-Centered
SAMPLE_TYPE = False  # False -> SALib, True -> SKOPT LHC-Centered
sample_nbr = 10  # number of intervals (sub-spaces) in each parameter range! or BINS NUMBER
if SAMPLE_TYPE:
    NbRuns = sample_nbr ** len(VarName2Change)
