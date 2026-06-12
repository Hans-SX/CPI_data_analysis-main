#%%
import numpy as np

import argparse
import os
from os.path import join
from joblib import Parallel, delayed

from CPI import readConfig, setDirectories_twocams
from utils import Calculating_G2, Timer, refocusing

#%%
exec(readConfig())
parser = argparse.ArgumentParser()
parser.add_argument('--DataSet', type=str)
parser.add_argument('--refName', nargs='?', default='refocused', type=str)
args = parser.parse_args()

datapath = join(os.getcwd(), os.pardir, args.DataSet, 'data')
outpath = join(os.getcwd(), os.pardir, args.DataSet, args.refName)
outDir, armAfiles, armBfiles = setDirectories_twocams(stdData=STD_PATH, stdOut=STD_PATH, timeTag=TT_BOOL, dataPath=datapath, outPath=outpath, armA=armA_PATH, armB=armB_PATH)

z_from_slope = np.load(join(outpath, 'z_from_slope.npz'))
avg_z = z_from_slope['avg_z']
# z_from_x = z_from_slope['z_from_x']
# z_from_y = z_from_slope['z_from_y']
# ref_pos = zip(avg_z, z_from_x, z_from_y)

timer = Timer()
timer.start("Whole refocusing")

cyc = 0
sigmas_2D = []
cov = []
total_iterations = len(armAfiles)

for Afile, Bfile, z in zip(armAfiles, armBfiles, avg_z):
    arr = Calculating_G2(Afile, Bfile, frames=50)
    G2 = arr.correlation(binA, binB)
    G2 = arr.padding(pad=25)
    
    # save_path = join(outDir, str(round(z)) + " mu m")
    save_path = join(outpath, 'refimages')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if cyc == 0:
        shape = G2.shape
        NA, NB= shape[0], shape[2]
        rangeA= (np.arange(NA)-(NA-1)/2)*dpA
        rangeB= (np.arange(NB)-(NB-1)/2)*dpB

    timer.start("Refocusing for z = " + str(round(z, 3)) + " mm")
    
    refVec = refocusing(G2, z, M_ratio, rangeA, rangeB, dpA, dpB, NA, NB, save_path, ind=cyc)
    # refs = Parallel(n_jobs=-1, backend="loky")(
        # delayed(refocusing)(G2, s, M_ratio, rangeA, rangeB, dpA, dpB, NA, NB, save_path, ind, maxInt) for ind, s in enumerate(z)
    # )    
    timer.stop("Refocusing for z = " + str(round(z, 3)) + " mm")
    
    if not os.path.exists(join(outpath, 'refVecs')):
        os.makedirs(join(outpath, 'refVecs'))
    # Since each refocused image has a different size, cannot save them in a single numpy array. Instead, save them as a list of arrays in a npy file. When loading, set allow_pickle=True to load the list of arrays and transform each element to a float such that it can be treated as an image.
    np.save(join(outpath, 'refVecs', str(cyc + 1) + "_" + str(round(z, 3)) + "mm" + "_refVec.npy"), np.array(refVec, dtype=object), allow_pickle=True)

    if (cyc+1) % 10 == 0:
        print("Iteration " + str(cyc+1) + " of " + str(total_iterations) + " finished.")

    cyc += 1

timer.stop("Whole refocusing")
timer.savefile(join(outpath, "refocusing_times.txt"))