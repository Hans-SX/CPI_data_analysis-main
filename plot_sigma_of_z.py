#%%
import csv
import sys
import numpy as np
import argparse
import os
from os.path import join
from joblib import Parallel, delayed
from re import search
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors

from utils import Timer, fit_gaussian_2D, kernel_smooth

"""
Fitting the refocused image to a 2D Gaussian, find the width of it and the center shifting in x and y when moving along z.
Inputs are the positions z and the corresponding refocused image file.
"""


def find_sigmas(imgpath):
    refVec = np.load(imgpath, allow_pickle=True)
    refVec = np.array(refVec, dtype=float)
    # gaussuan_2D_fit = fit_gaussian_2D(refVec, ls=True)
    gaussuan_2D_fit = fit_gaussian_2D(refVec, ls=False)
    sigmas = gaussuan_2D_fit[3]
    cov = gaussuan_2D_fit[5]
    center = gaussuan_2D_fit[1:3]
    center_guess = gaussuan_2D_fit[6]
    return sigmas, cov, center, center_guess, gaussuan_2D_fit[7]

"""
r = 25 * 1e-3      # radius of the pinhole, assume Gaussian pinhole, the transmittance.
f = 26.67           # focal length of the objective lens
k = 2 * np.pi / (532 * 1e-6)  # wave number of the laser, wavelength = 532 nm

The following three functions are the theoretical width of the refocused image when moving along z.

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
"""
#%%
parser = argparse.ArgumentParser()
parser.add_argument('--DataSet', type=str)
parser.add_argument('--refName', nargs='?', default='refocused', type=str)
parser.add_argument('--sigmaName', nargs='?', default='sigma_of_z', type=str)
args = parser.parse_args()

datapath = join(os.getcwd(), os.pardir, args.DataSet, 'data')
refpath = join(os.getcwd(), os.pardir, args.DataSet, args.refName)
outpath = join(refpath, args.sigmaName)

config_file = os.path.join(refpath, "config.py")

config = {}
with open(config_file) as f:
    exec(f.read(), config)

effect_size = config['binA'] * config['dpA']
# effect_size = 1


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
# np.save(join(outpath, "sig"), sigmas_2D)
sigmas_2D = sigmas_2D * effect_size
cov = np.array([res[i][1] for i in range(len(res))])
cov = cov * effect_size
coord = np.array([res[i][2] for i in range(len(res))])
# np.save(join(outpath, "coord"), coord)
coord = coord * effect_size
pcov = np.array([res[i][4] for i in range(len(res))])
pcov = pcov * effect_size
np.save(join(outpath, "pcov_analysis"), pcov)
# sys.exit()
err = np.sqrt(np.array(cov))
err_cx = err[:, 0]
err_cy = err[:, 1]
err_sig = err[:, 2]

np.savez(join(outpath, "width_analysis"), sigmas_2D=sigmas_2D, err_cx=err_cx, err_cy=err_cy, err_sig=err_sig, coord=coord)

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

# threshold = np.where(abs(sigmas_2D) < 6 * effect_size)
threshold = np.where(abs(err_sig) < 10)
sigmas = sigmas_2D[threshold]    # Set the threshold to 6 for a more clear visualization of the sigma of z. The threshold can be adjusted based on the specific data and analysis requirements.
avg_z_th = avg_z[threshold]
coord_th = coord[threshold]
err_cx = err_cx[threshold]
err_cy = err_cy[threshold]
err_sig = err_sig[threshold]


# Saturated frames.
sat = np.load(join(os.getcwd(), os.pardir, args.DataSet, r"saturated_analysis\n_saturated_frames.npz"), allow_pickle=True)

sat = sat['n_saturated_frames']             # The array recorded the num of saturated frames for each data point.
sat = sat[threshold]
frames = np.where(sat!=0)[0]                # Only data points with at least 1 satureated frame.
sat_z = avg_z_th[frames]
weights = sat[np.where(sat!=0)] * 100 / 50

norm = mcolors.Normalize(vmin=weights.min(), vmax=weights.max())        # Normalizing the weights.
cmap = plt.get_cmap('Reds')
cmap = mcolors.LinearSegmentedColormap.from_list(
    'reds_trim', cmap(np.linspace(0.15, 0.7, 256)))
strip_colors = cmap(norm(weights))

fig_width, ax_width = plt.subplots()

# ax_width.scatter(avg_z_th, abs(sigmas), label='Sigma fit')
# ax_width.fill_between(avg_z_th, abs(sigmas) - err_sig, abs(sigmas) + err_sig, color='C0', alpha=0.2, label='Error')
ax_width.errorbar(avg_z_th, abs(sigmas), yerr=err_sig, fmt='s', markersize=2, capsize=2)

ax_width.set_xlabel('Axial position (mm)')
ax_width.set_ylabel('Sigma of Gaussian fit (mm)')
ax_width.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
# ax2 = ax_width.twinx()
# ax2.errorbar(avg_z_th, abs(sigmas), yerr=err_sig, fmt='s', markersize=2, capsize=2)

ax_width.set_title('Estimated width.')
ax_width.vlines(sat_z, 0, 1, transform=ax_width.get_xaxis_transform(),
          colors=strip_colors, linewidth=8, zorder=0)
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
fig_width.colorbar(sm, ax=ax_width, label='Saturated frames in proportion (%)')
lines1, labels1 = ax_width.get_legend_handles_labels()
ax_width.legend(lines1, labels1, loc='lower right')
plt.savefig(join(outpath, "estimated_width"), dpi='figure', transparent=False)

time = range(len(coord_th[:, 0]))
# r = kernel_smooth(time, coord_th[:, 0], yerr=err_cx, bw=4.0)
tg, m, s, w = kernel_smooth(time, coord_th[:, 0], bw=4.0)

fig, ax1 = plt.subplots()
# fig, ax1 = plt.subplots(figsize=(8,5))
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Center coordinates of Gaussian fit (mm)')
ax1.errorbar(range(coord_th[:,0].shape[0]), coord_th[:,0], yerr=err_cx, fmt='s', markersize=2, capsize=2, label='Center x')
# ax1.scatter(range(coord_th[:,0].shape[0]), coord_th[:,0], label='Center x')

# ax1.plot(r["t"], r["mean"], "k-", lw=2, label="Weighted mean", zorder=5)
# ax1.fill_between(r["t"], r["mean"]-r["spread"], r["mean"]+r["spread"],
#                 color="k", alpha=0.15, label="±1σ fluctuation")
# ax1.fill_between(r["t"], r["mean"]-r["sem"], r["mean"]+r["sem"],
#                 color="C3", alpha=0.5, label="±1σ of mean")
ax1.plot(tg, m, color="k", lw=2, label="Smoothed mean", zorder=5)
ax1.fill_between(tg, m - s, m + s, color="k", alpha=0.2,
                label="±1σ fluctuation", zorder=4)

ax1.grid(True, alpha=0.3)
ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
# ax2 = ax1.twinx()
# ax2.set_ylabel('Axial position (mm)')
# ax2.plot(time_platf, pos_platf, label='Platform position')

lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1, labels1, loc='lower right')
# ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')


ax1.vlines(frames, 0, 1, transform=ax1.get_xaxis_transform(),
          colors=strip_colors, linewidth=8, zorder=0)
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
fig.colorbar(sm, ax=ax1, label='Saturated frames in proportion (%)')
# fig.tight_layout()
plt.savefig(join(outpath, "Center_x_of_sigma.png"), dpi='figure', transparent=False)
plt.close("all")


time = range(len(coord_th[:, 1]))
# r = kernel_smooth(time, coord_th[:, 1], yerr=err_cy, bw=4.0)
tg, m, s, w = kernel_smooth(time, coord_th[:, 1], bw=4.0)

fig, ax1 = plt.subplots()
# fig, ax1 = plt.subplots(figsize=(8,5))
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Center coordinates of Gaussian fit (mm)')
ax1.errorbar(range(coord_th[:,1].shape[0]), coord_th[:,1], yerr=err_cy, fmt='s', markersize=2, capsize=2, label='Center y')
# ax1.scatter(range(coord_th[:,1].shape[0]), coord_th[:,1], label='Center y')
# ax1.plot(r["t"], r["mean"], "k-", lw=2, label="Weighted mean", zorder=5)
# ax1.fill_between(r["t"], r["mean"]-r["spread"], r["mean"]+r["spread"],
#                 color="k", alpha=0.15, label="±1σ fluctuation")
# ax1.fill_between(r["t"], r["mean"]-r["sem"], r["mean"]+r["sem"],
#                 color="C3", alpha=0.5, label="±1σ of mean")
ax1.plot(tg, m, color="k", lw=2, label="Smoothed mean", zorder=5)
ax1.fill_between(tg, m - s, m + s, color="k", alpha=0.2,
                label="±1σ fluctuation", zorder=4)

ax1.grid(True, alpha=0.3)
ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
# ax2 = ax1.twinx()
# ax2.set_ylabel('Axial position (mm)')
# ax2.plot(time_platf, pos_platf, label='Platform position')

lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1, labels1, loc='lower right')
# ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')


ax1.vlines(frames, 0, 1, transform=ax1.get_xaxis_transform(),
          colors=strip_colors, linewidth=8, zorder=0)
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
fig.colorbar(sm, ax=ax1, label='Saturated frames in proportion (%)')
# fig.tight_layout()
plt.savefig(join(outpath, "Center_y_of_sigma.png"), dpi='figure', transparent=False)
plt.close("all")
