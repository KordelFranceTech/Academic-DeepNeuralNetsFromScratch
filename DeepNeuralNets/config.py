# config.py
# Kordel France
########################################################################################################################
# This file contains hyperparameters to control debugging features
########################################################################################################################

# set true if you want to see a verbose data log to diagnose issues
DEBUG_MODE = False
DEMO_MODE = False
# limit on how many values you want to limit processing to
DATA_LIMIT: int = 1000000
# directory name storing all of the i/o files
IO_DIRECTORY = 'Lab4/io_files'
# demo data for diagnostics and testing of algorithms
DEMO_DATA_X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
DEMO_DATA_Y = [2786, 2814, 2842, 2884, 2926, 3003, 3059, 3031, 3066, 3115]
