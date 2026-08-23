import argparse
import os
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import csv

from utils import Calculating_G2, ridge_fit, ridge_odr_fit, plot_ridge_fit
from CPI import readConfig, setDirectories_twocams
from temp import BigStepForward_SmallStepBack, mean_positions_per_second
from config import shift, binA, binB, M_err, z_of_slope

"""
This script is used to find the position of the object by fitting the slope of the 2D correlation function (xaxb or yayb) without refocusing.
"""

exec(readConfig())

parser = argparse.ArgumentParser()
parser.add_argument('--DataSet', type=str)
parser.add_argument('--refName', nargs='?', default='refocused', type=str)
args = parser.parse_args()

datapath = join(os.getcwd(), os.pardir, args.DataSet, 'data')
outpath = join(os.getcwd(), os.pardir, args.DataSet, args.refName)
outDir, armAfiles, armBfiles = setDirectories_twocams(stdData=STD_PATH, stdOut=STD_PATH, timeTag=TT_BOOL, dataPath=datapath, outPath=outpath, armA=armA_PATH, armB=armB_PATH)

if not os.path.exists(join(outDir, 'fit_plots')):
    os.makedirs(join(outDir, 'fit_plots'))

pattern = mean_positions_per_second(shift, BigStepForward_SmallStepBack(6.8, 12.8, pattern=np.array((6, -3))).pos_frames(interval=0.1, time_interval=50)['pos'], speed=0.1)
try_ref_to = pattern[0]
expect_ref = pattern[1]

avg_z = []
err_z = []
z_from_x = []
z_from_y = []
err_from_x = []
err_from_y = []
# bt_z = []
# err_bt_z = []
cyc = 0
for Afile, Bfile, exp in zip(armAfiles, armBfiles, expect_ref):
# for Afile, Bfile in zip(armAfiles, armBfiles):
    arr = Calculating_G2(Afile, Bfile, frames=50)
    G2 = arr.correlation(binA, binB)

    xaxb = np.sum(G2, axis=(1,3))
    yayb = np.sum(G2, axis=(0,2))
    xaxb[np.where(xaxb < 0)] = 0
    yayb[np.where(yayb < 0)] = 0

    # res_x = ridge_fit(xaxb, intensity_power=5)
    # res_y = ridge_fit(yayb, intensity_power=5)
    res_x = ridge_odr_fit(xaxb, intensity_power=5)
    res_y = ridge_odr_fit(yayb, intensity_power=5)

    slope_x = - res_x['slope']
    slope_y = res_y['slope']
    err_x = - res_x['slope_err']
    err_y = res_y['slope_err']
    # if err_x < err_y:
    #     weighted = slope_x
    #     weighted_err = err_x
    # else:
    #     weighted = slope_y
    #     weighted_err = err_y
    weighted = (slope_x/err_x**2 + slope_y/err_y**2) / (1/err_x**2 + 1/err_y**2)
    weighted_err = np.sqrt(1/(1/err_x**2 + 1/err_y**2))

    err_z.append(np.sqrt((z_of_slope(weighted_err))**2 + (weighted * M_err)**2))
    avg_z.append(z_of_slope(weighted))
    z_from_x.append(z_of_slope(slope_x))
    z_from_y.append(z_of_slope(slope_y))
    err_from_x.append(z_of_slope(err_x))
    err_from_y.append(z_of_slope(err_y))

    # bt_x = z_of_slope(- res_x['slope_bt_mean'])
    # bt_y = z_of_slope(res_y['slope_bt_mean'])
    # err_bt_x = z_of_slope(- res_x['slope_bt_std'])
    # err_bt_y = z_of_slope(res_y['slope_bt_std'])
    # err_bt_z.append(np.sqrt(err_bt_x**2 + err_bt_y**2) / 2)
    # bt_z.append((bt_x + bt_y) / 2)


    fig, (ax1,ax2) = plt.subplots(2,figsize=(6,10))

    plot_ridge_fit(res_x, xaxb, ax=ax1)
    plot_ridge_fit(res_y, yayb, ax=ax2)
    fig.tight_layout()
    fig.savefig(join(outDir, 'fit_plots', str(cyc + 1) + f"_expected_z_{exp:03f}_mm.png"), dpi='figure',
    transparent=False)
    plt.close("all")

    cyc += 1

pos_platf = []
time_platf = []
with open(join(os.getcwd(), os.pardir, args.DataSet, 'positions.csv'), 'r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header row
    for row in reader:
        pos_platf.append(float(row[1]))
        time_platf.append(float(row[0]))
pos_platf = np.array(pos_platf, dtype=float) - 8.8 # 8.8 is the position of the focal plane on the platform measure.
time_platf = np.array(time_platf, dtype=float)

fig = plt.figure()
plt.plot(range(1, 1+len(expect_ref)), expect_ref, color='black', label='Expected Z')
plt.plot(time_platf, pos_platf, color='orange', label='Platform Position')
# plt.scatter(range(1, 1+len(avg_z)), avg_z, label='Average Z from Slope')
plt.errorbar(range(1, 1+len(avg_z)), avg_z, yerr=err_z, fmt='o', markersize=2, label='Average Z with Error Bar')
# plt.errorbar(range(1, 1+len(bt_z)), bt_z, yerr=err_bt_z, fmt='s', markersize=2, label='Bootstrap Z with Error Bar')

plt.title('Axial Position Analysis')
plt.legend()
fig.savefig(join(outDir, "z_from_wavg_slope.png"), dpi='figure', transparent=False)
plt.close("all")
np.savez(join(outDir, "z_from_slope.npz"), avg_z=avg_z, err_z=err_z, z_from_x=z_from_x, z_from_y=z_from_y, err_from_x=err_from_x, err_from_y=err_from_y)
# np.savez(join(outDir, "z_from_slope.npz"), avg_z=avg_z, err_z=err_z, bt_z=bt_z, err_bt_z=err_bt_z)