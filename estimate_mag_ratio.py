import numpy as np

import argparse
import os
from os.path import join
from joblib import Parallel, delayed
from scipy.optimize import curve_fit

from utils import cubic, get_m_min_and_error
from config import M_ratio

"""
This script is used to find the optimal magnification ratio by averaging the fitted values of sigma, where sigma is treated as a function of the magnification ratio. The sigma values are obtained from width_of_pos_ratio.py at different axial positions.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--DataSet', type=str)
parser.add_argument('--refName', type=str)
args = parser.parse_args()

datapath = join(os.getcwd(), os.pardir, args.DataSet, args.refName, 'sigmas.npy')

sigmas = np.load(datapath)

results = Parallel(n_jobs=-1, backend="loky")(
    delayed(curve_fit)(cubic, np.array(M_ratio), sigma) for sigma in sigmas
)

m_mins, m_errs = [], []
for popt, pcov in results:
    m_min, err = get_m_min_and_error(popt, pcov, M_ratio)
    m_mins.append(m_min)
    m_errs.append(err)

# Weighted average
m_mins = np.array(m_mins)
m_errs = np.array(m_errs)

valid = np.isfinite(m_mins) & np.isfinite(m_errs)
weights = 1.0 / (m_errs[valid]**2)
m_optimal = np.sum(weights * m_mins[valid]) / np.sum(weights)

print(f"Optimal ratio across all z: {m_optimal:.3f}")


