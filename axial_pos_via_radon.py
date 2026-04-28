import argparse
import os
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import radon

from utils import setDirectories_twocams, Calculating_G2
from CPI import readConfig

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
for Afile, Bfile in zip(armAfiles, armBfiles):
# for Afile, Bfile in zip(armAfiles, armBfiles):
    arr = Calculating_G2(Afile, Bfile, frames=100)
    G2 = arr.correlation(binA, binB)
    G2 = arr.padding(pad=25)

    # Apply Radon transform to the correlation function
    theta = np.linspace(0., 180., max(G2.shape), endpoint=False)
    sinogram = radon(np.sum(G2, axis=(1,3)), theta=theta, circle=False)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(G2, cmap='gray')
    ax2.imshow(sinogram, cmap='gray')
    plt.show()
    cyc += 1
