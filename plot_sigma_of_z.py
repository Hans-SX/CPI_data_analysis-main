#%%
import csv

import numpy as np
import argparse
import os
from os.path import join
from joblib import Parallel, delayed
from re import search
import matplotlib.pyplot as plt

from utils import Timer, fit_gaussian_2D

def find_sigmas(imgpath):
    refVec = np.load(imgpath, allow_pickle=True)
    refVec = np.array(refVec, dtype=float)
    gaussuan_2D_fit = fit_gaussian_2D(refVec, ls=True)
    sigmas = gaussuan_2D_fit[3]
    cov = gaussuan_2D_fit[5]
    center = gaussuan_2D_fit[1:3]
    center_guess = gaussuan_2D_fit[6]
    return sigmas, cov, center, center_guess

r = 2.5 * 1e-3      # radius of the pinhole, assume Gaussian pinhole, the transmittance.
f = 26.67           # focal length of the objective lens
k = 2 * np.pi / 0.532  # wave number of the laser, wavelength = 532 nm

def sigma_ideal(z):
    # l -> infinity; sigma_c -> 0
    return np.sqrt(z**2 / (2 * k**2 * r**2) + r**2 / 2)

def sigma_coh(z, sigma_c):
    # l -> infinity;
    return np.sqrt(z**2 / (2 * k**2 * r**2) + r**2 / 2 * ((1 + sigma_c**2 / (2 * r**2)) / (1 + sigma_c**2 / (4 * r**2))))

def sigma_lens(z, l):
    # sigma_c -> 0
    l_term = (f**2 + z**2) / (4 * k**2 * l**2)
    r_term = r**2 / 2
    z_term = ((1 + f/z)**2 / (2 * l**2) + (2 * k**2 * r**2) / z**2)**(-1)
    return np.sqrt(l_term + r_term + z_term)

#%%
parser = argparse.ArgumentParser()
parser.add_argument('--DataSet', type=str)
parser.add_argument('--refName', nargs='?', default='refocused', type=str)
parser.add_argument('--sigmaName', nargs='?', default='sigma_of_z', type=str)
args = parser.parse_args()

datapath = join(os.getcwd(), os.pardir, args.DataSet, 'data')
refpath = join(os.getcwd(), os.pardir, args.DataSet, args.refName)
outpath = join(refpath, args.sigmaName)

z_from_slope = np.load(join(refpath, 'z_from_slope.npz'))
refVecpath = join(refpath, 'refVecs')
reflist = [f for f in os.listdir(refVecpath) if f.endswith('_refVec.npy')]
# Sort numerically by the number at the start of the filename
reflist.sort(key=lambda f: int(search(r'(\d+)', f).group(1)))
reflist = [join(refpath, 'refVecs', f) for f in reflist]

if not os.path.exists(outpath):
    os.makedirs(outpath)
avg_z = z_from_slope['avg_z']
z_from_x = z_from_slope['z_from_x']
z_from_y = z_from_slope['z_from_y']
# ref_pos = zip(avg_z, z_from_x, z_from_y)

timer = Timer()
timer.start("Sigma analysis of refocused images")

res = Parallel(n_jobs=-1, backend="loky")(delayed(find_sigmas)
                            (imgpath) for imgpath in reflist)

timer.stop("Sigma analysis of refocused images")
sigmas_2D = np.array([res[i][0] for i in range(len(res))])
cov = np.array([res[i][1] for i in range(len(res))])
coord = np.array([res[i][2] for i in range(len(res))])
# coord = np.array([res[i][3] for i in range(len(res))])
err = np.sqrt(np.array(cov))
np.save(join(outpath, "sigmas_2D"), sigmas_2D)
np.save(join(outpath, "err"), err)
np.save(join(outpath, "coord"), coord)
# np.save(join(outpath, "center_guess"), center_guess)

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

threshold = np.where(abs(sigmas_2D) < 6)
sigmas = sigmas_2D[threshold]    # Set the threshold to 6 for a more clear visualization of the sigma of z. The threshold can be adjusted based on the specific data and analysis requirements.
avg_z_th = avg_z[threshold]
coord_th = coord[threshold]
fig = plt.figure()
# plt.errorbar(
#     avg_z, abs(sigmas_2D), yerr=err,
#     fmt='o',          # scatter-like markers (no connecting line if you set linestyle)
#     capsize=4,        # add caps on error bars
#     ecolor='black',   # error bar color
#     elinewidth=1.5,   # error bar line width
#     markersize=2,
#     label='Data ± error'
# )
plt.scatter(avg_z_th, abs(sigmas), label='Sigma of weighted avg z.')
plt.xlabel('Position (mm)')
plt.ylabel('Sigma of Gaussian fit (pixels)')
plt.title('Sigma of z.')
plt.legend()
plt.savefig(join(outpath, "sigma_of_z.png"), dpi='figure', transparent=False)
plt.close("all")

fig, ax1 = plt.subplots(figsize=(8,5))
ax1.set_xlabel('Time order')
ax1.set_ylabel('Center coordinates of Gaussian fit (pixel)')
ax1.scatter(range(coord_th[:,0].shape[0]), coord_th[:,0], label='Center x')

ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.set_ylabel('Expect positions')
ax2.plot(time_platf, pos_platf, label='Platform Z')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

fig.tight_layout()
plt.savefig(join(outpath, "Center_x_of_sigma.png"), dpi='figure', transparent=False)
plt.close("all")

fig, ax1 = plt.subplots(figsize=(8,5))
ax1.set_xlabel('Time order')
ax1.set_ylabel('Center coordinates of Gaussian fit (pixel)')
ax1.scatter(range(coord_th[:,1].shape[0]), coord_th[:,1], label='Center y', color='orange')

ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.set_ylabel('Expect positions')
ax2.plot(time_platf, pos_platf, label='Platform Z')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

fig.tight_layout()
plt.savefig(join(outpath, "Center_y_of_sigma.png"), dpi='figure', transparent=False)
plt.close("all")
