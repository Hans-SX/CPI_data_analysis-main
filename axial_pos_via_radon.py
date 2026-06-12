import argparse
import os
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import radon

from utils import Calculating_G2, line
from CPI import readConfig, setDirectories_twocams
from temp import BigStepForward_SmallStepBack, mean_positions_per_second
from config import shift, binA, binB, z_of_slope

"""
There is an error depending on the ratio of the size of the object and the size of the lens. Since the size of the object is unknown, the ratio is also unknown, which makes it hard to apply the correction term.
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

cyc = 0
for Afile, Bfile, exp in zip(armAfiles, armBfiles, expect_ref):
# for Afile, Bfile in zip(armAfiles, armBfiles):
    arr = Calculating_G2(Afile, Bfile)
    G2 = arr.correlation(binA, binB)
    
    xaxb = np.sum(G2, axis=(1,3))
    yayb = np.sum(G2, axis=(0,2))

    # Apply Radon transform to the correlation function
    theta = np.linspace(-90., 90., 540, endpoint=False)
    sinogram_x = radon(xaxb, theta=theta, circle=False)
    sinogram_y = radon(yayb, theta=theta, circle=False)
    pos_x, phi_x = np.unravel_index(np.argmax(sinogram_x), sinogram_x.shape)
    pos_y, phi_y = np.unravel_index(np.argmax(sinogram_y), sinogram_y.shape)
    ang_x = theta[phi_x]
    ang_y = theta[phi_y]
    slope_x = np.tan(np.deg2rad(ang_x))
    slope_y = np.tan(np.deg2rad(ang_y))
    x_line = line(range(xaxb.shape[1]), slope_x, 120)
    y_line = line(range(yayb.shape[1]), slope_y, 0)
    z_from_x = z_of_slope(slope_x)
    z_from_y = z_of_slope(slope_y)

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
plt.scatter(range(1, 1+len(expect_ref)), avg_z, label='Average Z from Slope')
plt.title('Axial Position Analysis')
plt.legend()
fig.savefig(join(outDir, "z_from_slope.png"), dpi='figure', transparent=False)
plt.close("all")