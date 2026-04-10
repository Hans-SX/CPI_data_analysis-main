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
binA, binB = 2, 2
dpA, dpB   = 6.5*4, 6.5*4
# REFOCUSING
M_ratio = range(1, 21, 1)

maxInt = False

# Array of planes to be refocused
REFOC = range(45, 127, 2)

def transf(z, M_ratio):
    from numpy import array as npArray
    f=26670.
    # f=30000.
    return npArray([
        [- 1, - z/f],
        [0         ,       -1/M_ratio * dpA * binA /dpB /binB]
        ])


# Do you want to apply the correction term?
CORREC_BOOL = False