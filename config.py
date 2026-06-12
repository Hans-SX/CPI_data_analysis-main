# Selecting folders
STD_PATH   = False             # True for using standard folders
                              # False for user-defined folders listed below
armA_PATH = 'spatial'
armB_PATH = 'angular'

# Time tag output folder?
TT_BOOL = False

#   Definition of folders
DATA_DIR   = ""
OUTPUT_DIR = ""
G2_DIR     = ""

# Binning
binA, binB = 4, 4
dpA, dpB   = 6.5* 1e-3 *4, 6.5* 1e-3 *4
# REFOCUSING
# M_ratio = range(1, 21, 1)
M_ratio = 12.682
M_err = 0.002

maxInt = False

pixA, pixB = dpA*binA, dpB*binB
focal = 26.67
dis = focal
# Array of planes to be refocused
# REFOC = range(45, 127, 2)

def transf(z, M_ratio):
    import numpy as np
    from numpy import array as npArray
    # focal=30000.
    return npArray([
        [- 1, - z/focal],
        [0         ,       -1/M_ratio * dpA * binA /dpB /binB]
        ], dtype=np.float32)

def shift(position):
    # for swiping angular arm alignment error on FPP:
    shift = - position / focal * M_ratio / (1 + position / focal * (1 - dis/focal)) * pixB/pixA
    # for swiping ratios: shift = - position * M_ratio/ (position + focal) * pixB/pixA
    # for the simulation data: position * MA/MB / (position + focal) * pixB/pixA
    return shift

def z_of_slope(slope):
    return - slope * focal * dpA * binA / (dpB * binB * M_ratio)

# Do you want to apply the correction term?
CORREC_BOOL = False