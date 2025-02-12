# Selecting folders
STD_PATH   = False             # True for using standard folders
                              # False for user-defined folders listed below
armA_PATH = 'spatial'
armB_PATH = 'angular'

# Time tag output folder?
TT_BOOL = False

#   Definition of folders
DATA_DIR   = ""
OUTPUT_DIR = "1_angBin2"
G2_DIR     = ""

# Cordinates for data cropping
Na = 250;  ax = 208;    ay = 275
Nb = 360;  bx = 120;    by = 1231

# Binning
(binA, binB)=(2, 2)
dp = 6.5
# REFOCUSING
Ma = 4.2
Mb = 0.32
maxInt = False

# Array of planes to be refocused
REFOC = range(28000, 30000, 100)
# REFOC = range(30600, 31650, 50)
# REFOC = [28900, 29000, 29100, 30900, 31000, 31100]
# REFOC = [29000, 29100]

# Do you want differential?
DIFF_BOOL = True  # Previously False.

def transf(z):
    from numpy import array as npArray
    f=30000.
    return npArray([
        [-f/z*Ma,(f/z-1)*Ma],
        [0         ,       -Mb]
        ])

# Do you want to apply the correction term?
CORREC_BOOL = False