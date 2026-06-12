import argparse
import os
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from utils import Calculating_G2, line
from CPI import readConfig, setDirectories_twocams
from temp import BigStepForward_SmallStepBack, mean_positions_per_second
from config import shift, binA, binB, z_of_slope

"""
Mainly for figure out the relationship between the axial position and the slope of the correlation function, it serves as a baseline to better understand the radon transform method. This is a preliminary analysis to track the axial position without refocusing, which would be much faster.
"""

exec(readConfig())

parser = argparse.ArgumentParser()
parser.add_argument('--DataSet', type=str)
parser.add_argument('--refName', nargs='?', default='refocused', type=str)
args = parser.parse_args()

datapath = join(os.getcwd(), os.pardir, args.DataSet, 'data')
outpath = join(os.getcwd(), os.pardir, args.DataSet, args.refName)
outDir, armAfiles, armBfiles = setDirectories_twocams(stdData=STD_PATH, stdOut=STD_PATH, timeTag=TT_BOOL, dataPath=datapath, outPath=outpath, armA=armA_PATH, armB=armB_PATH)

pattern = mean_positions_per_second(shift, BigStepForward_SmallStepBack(6.8, 12.8, pattern=np.array((6, -3))).pos_frames(interval=0.1, time_interval=50)['pos'], speed=0.1)
try_ref_to = pattern[0]
expect_ref = pattern[1]

avg_z = []
cov = []
cyc = 0
for Afile, Bfile, exp in zip(armAfiles, armBfiles, expect_ref):
# for Afile, Bfile in zip(armAfiles, armBfiles):
    arr = Calculating_G2(Afile, Bfile, frames=50)
    G2 = arr.correlation(binA, binB)

    xaxb = np.sum(G2, axis=(1,3))
    yayb = np.sum(G2, axis=(0,2))

    x_val, x_domain = np.where(xaxb > 0.3 * np.max(xaxb))
    y_val, y_domain = np.where(yayb > 0.3 * np.max(yayb))

    x_popt, x_pcov = curve_fit(line, x_domain, x_val)
    y_popt, y_pcov = curve_fit(line, y_domain, y_val)

    x_line = line(range(xaxb.shape[1]), *x_popt)
    y_line = line(range(yayb.shape[1]), *y_popt)

    z_from_x = z_of_slope(- x_popt[0])
    z_from_y = z_of_slope(y_popt[0])
    cov_x = z_of_slope(- x_pcov[0, 0])
    cov_y = z_of_slope(y_pcov[0, 0])

    avg_z.append((z_from_x + z_from_y) / 2)
    cov.append((np.sqrt(cov_x) + np.sqrt(cov_y)) / 2)

    fig, (ax1,ax2) = plt.subplots(2,figsize=(6,10))
    im1 = ax1.imshow(xaxb, cmap="gray")
    ax1.plot(x_line, color="red")
    im2 = ax2.imshow(yayb, cmap="gray")
    ax2.plot(y_line, color="red")
    ax1.set_title(f"xA-xB Corr. Func., z from xaxb: {round(z_from_x, 3)} mm")
    ax2.set_title(f"yA-yB Corr. Func., z from yayb: {round(z_from_y, 3)} mm")
    fig.savefig(join(outDir, str(cyc + 1) + f"_expected_z_{exp:03f}_mm.png"), dpi='figure',
    transparent=False)
    plt.close("all")

    cyc += 1

fig = plt.figure()
plt.plot(range(1, 1+len(expect_ref)), expect_ref, color='black', label='Expected Z')
# plt.scatter(range(1, 1+len(expect_ref)), avg_z, label='Average Z from Slope')
plt.errorbar(range(1, 1+len(expect_ref)), avg_z, yerr=cov, fmt='o', markersize=2, label='Average Z with Error Bar')
plt.title('Axial Position Analysis')
plt.legend()
fig.savefig(join(outDir, "z_from_slope.png"), dpi='figure', transparent=False)
plt.close("all")

np.savez(join(outDir, "z_from_slope.npz"), avg_z=avg_z, cov=cov)