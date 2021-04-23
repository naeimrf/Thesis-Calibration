import sys, os
"""
This script is a central place to hold initial parameters for all local scripts!
"""
# add the required path
path2addgeom = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'geomeppy')
sys.path.append(path2addgeom)

# OPTION 1: ALWAYS KEEP RECURSIVE_CALIBRATION = False <-
# 1. FIRST RUN
#   -> SIMULATE_TO_CALIBRATE = True AND CALIBRATE_WITH_CALIBRATED_PARAMETERS = False THEN RUN local_builder.py
# - - - - - - - - - - - - - - -
# 2. TO CALIBRATE PARAMETERS
#   -> SIMULATE_TO_CALIBRATE = False AND CALIBRATE_WITH_CALIBRATED_PARAMETERS = False THEN RUN local_calib.py
# - - - - - - - - - - - - - - -
# 3. TO SIMULATE WITH CALIBRATED PARAMETERS
#   -> SIMULATE_TO_CALIBRATE = True AND CALIBRATE_WITH_CALIBRATED_PARAMETERS = True THEN RUN local_builder.py

SIMULATE_TO_CALIBRATE = False
CALIBRATE_WITH_CALIBRATED_PARAMETERS = False

# OPTION 2: SET THE BOOLEAN BELOW TO 'TRUE' TO RUN 'local_builder' RECURSIVELY,
# UNTIL ALL BUILDINGS CAN SATISFY BETA VALUE IN CALIBRATION ALGORITHM!
# THIS OPTION IS SUITABLE FOR SEVERAL-TIME CALIBRATIONS BUT WITH LOW NUMBER OF SIMULATIONS!
RECURSIVE_CALIBRATION = True


CaseName = 'ForTest_6_20'  # a folder_name to save input and outfile in form of pickles!
BuildNum = [6]  # building numbers (names) to be simulated  #  [i for i in range(7, 28)]
# Before each run, pay attention to Outputs.txt (lines: 110, 353, 359)
VarName2Change = ['EnvLeak', 'wwr', 'IntLoadMultiplier', 'AreaBasedFlowRate', 'setTempLoL']
Bounds = [[0.4, 2], [0.2, 0.4], [0.5, 2], [0.3, 0.6], [19, 22]]

# Morris analysis needs 'NbRuns' to be a multiplier of 'number of parameters'
# That is if you have 5 parameters you need (5+1) * k number simulations. Ex: 17x6=102
NbRuns = 20

# ** EXPERIMENTAL CODE **
# SALib or SKOPT LHC-Centered
SAMPLE_TYPE = False  # False -> SALib, True -> SKOPT LHC-Centered
sample_nbr = 5  # number of intervals (sub-spaces) in each parameter range!
if SAMPLE_TYPE:
    NbRuns = sample_nbr ** len(VarName2Change)
