#%%
import numpy as np

import argparse
import os
from os.path import join
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

from CPI import readConfig, setDirectories_twocams
from utils import Calculating_G2, Timer, refocusing, robust_gaussian_fit, plt_sigma
exec(readConfig())
from temp import BigStepForward_SmallStepBack, mean_positions_per_second
from config import shift

parser = argparse.ArgumentParser()
parser.add_argument('--DataSet', type=str)
parser.add_argument('--refName', nargs='?', default='refocused', type=str)
args = parser.parse_args()

datapath = join(os.getcwd(), os.pardir, args.DataSet, 'data')
outpath = join(os.getcwd(), os.pardir, args.DataSet, args.refName)
outDir, armAfiles, armBfiles = setDirectories_twocams(stdData=STD_PATH, stdOut=STD_PATH, timeTag=TT_BOOL, dataPath=datapath, outPath=outpath, armA=armA_PATH, armB=armB_PATH)


timer = Timer()
timer.start("Whole refocusing")

cyc = 0
sigmas_x = []
sigmas_y = []

# REFOC = (np.array(REFOC) * 0.1 - 8.5) * 1000  # convert to micro meters and shift the origin to the position of the platform when z=0, which is 8.5 mm in the original scale.
pattern = mean_positions_per_second(shift, BigStepForward_SmallStepBack(6.8, 12.8, pattern=np.array((6, -3))).pos_frames(interval=0.1, time_interval=50)['pos'], speed=0.1)
try_ref_to = pattern[0]
expect_ref = pattern[1]
total_iterations = len(armAfiles)
#%%
# for Afile, Bfile, z in zip(armAfiles, armBfiles, REFOC):
for Afile, Bfile, z, exp in zip(armAfiles, armBfiles, try_ref_to, expect_ref):
    arr = Calculating_G2(Afile, Bfile, frames=50)
    G2 = arr.correlation(binA, binB)
    G2 = arr.padding(pad=25)
    
    # save_path = join(outDir, str(round(z)) + " mu m")
    save_path = join(outDir, str(cyc + 1) + "_" + str(round(exp, 3)) + "mm")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if cyc == 0:
        shape = G2.shape
        NA, NB= shape[0], shape[2]
        rangeA= (np.arange(NA)-(NA-1)/2)*dpA
        rangeB= (np.arange(NB)-(NB-1)/2)*dpB

    # refs = Parallel(n_jobs=-1, backend="loky")(
    #     delayed(refocusing)(G2, z, mr, rangeA, rangeB, dpA, dpB, NA, NB, save_path, maxInt) for mr in M_ratio
    # )
    refs = Parallel(n_jobs=-1, backend="loky")(
        delayed(refocusing)(G2, s, M_ratio, rangeA, rangeB, dpA, dpB, NA, NB, save_path, ind, maxInt) for ind, s in enumerate(z)
    )

    refVec = [x for x in refs]
    if not os.path.exists(join(outDir, 'refVecs')):
        os.makedirs(join(outDir, 'refVecs'))
    np.save(join(outDir, 'refVecs', str(cyc + 1) + "_" + str(round(exp, 3)) + "mm" + "_refVec.npy"), np.array(refVec, dtype=object), allow_pickle=True)

    sum_refVec = [np.sum(ref, axis=1) for ref in refVec]
    argmax_sum = [np.argmax(s) for s in sum_refVec]

    gaussian_fits_x = Parallel(n_jobs=-1, backend="loky")(
        delayed(robust_gaussian_fit)(np.arange(ref.shape[0]), np.sum(ref, axis=1)) for ref in refVec
    )
    gaussian_fits_y = Parallel(n_jobs=-1, backend="loky")(
        delayed(robust_gaussian_fit)(np.arange(ref.shape[0]), np.sum(ref, axis=0)) for ref in refVec
    )
    # gaussian_fits is a list of parallel results, each element is a tuple of (popt, pcov) for the corresponding refocused image.
    sigma_x = np.array([fit[2] if fit is not None else np.nan for fit in gaussian_fits_x])  # shape: (len(M_ratio),)
    sigma_y = np.array([fit[2] if fit is not None else np.nan for fit in gaussian_fits_y])  # shape: (len(M_ratio),)

    # print("sigma for z = {}: {}".format(z, sigma_x))
    plt_sigma(M_ratio, sigma_x, z, outpath, cyc+1, axis='x')
    plt_sigma(M_ratio, sigma_y, z, outpath, cyc+1, axis='y')


    if (cyc+1) % 10 == 0:
        print("Iteration " + str(cyc+1) + " of " + str(total_iterations) + " finished.")
    sigmas_x.append(sigma_x)
    sigmas_y.append(sigma_y)

    cyc += 1

argmin_sig_x = np.argmin(np.array(sigmas_x), axis=1)
argmin_sig_y = np.argmin(np.array(sigmas_y), axis=1)
best_pos_x = np.array(try_ref_to)[np.arange(np.array(try_ref_to).shape[0]), argmin_sig_x]
best_pos_y = np.array(try_ref_to)[np.arange(np.array(try_ref_to).shape[0]), argmin_sig_y]
best_pos = (best_pos_x + best_pos_y) / 2
fig = plt.figure()
plt.plot(best_pos_x, label='Best Position from X')
plt.plot(best_pos_y, label='Best Position from Y')
plt.plot(best_pos, label='Best Position (Average)')

plt.xlabel("Interval")
plt.ylabel("Axial position (mm)")
plt.title("Trend of best axial position across intervals")
plt.legend()

fig.savefig(join(outDir, "trend.png"), dpi='figure',transparent=False)
plt.close("all")
# sigmas = np.array(sigmas)  # shape: (len(REFOC), len(M_ratio))
#%%
np.save(join(outDir, "sigmas_x.npy"), np.array(sigmas_x))
np.save(join(outDir, "sigmas_y.npy"), np.array(sigmas_y))
np.save(join(outDir, "best_pos_x.npy"), np.array(best_pos_x))
np.save(join(outDir, "best_pos_y.npy"), np.array(best_pos_y))
np.save(join(outDir, "best_pos.npy"), np.array(best_pos))

timer.stop("Whole refocusing")
