import matplotlib.pyplot as plt
import os
import time
from tifffile import imread
import shutil
import itertools
from os.path import join

import numpy as np
from scipy.optimize  import curve_fit

import CPI as cpi
from CPI import PrintSectionInit, PrintSectionClose, Print, DataDir, joinDir, OutDir
from scipy.interpolate import interpn as interp
from config import transf


def cubic(m, a, b, c, d):
    return a + b*m + c*m**2 + d*m**3

def get_m_min_and_error(popt, pcov, M_ratio):
    """
    Extract minimum ratio m_min and its error from cubic fit.
    
    Args:
        popt: [a, b, c, d] from cubic fit
        pcov: covariance matrix from curve_fit
        M_ratio: array of m ratios used in fit (for bounds checking)
    
    Returns:
        (m_min, delta_m_min) or (nan, nan) if no valid minimum
    """
    _, b, c, d = popt
    
    # Solve sigma'(m) = b + 2c*m + 3d*m^2 = 0
    coeffs = [3*d, 2*c, b]
    roots = np.roots(coeffs)
    
    # Find physical minimum
    m_min = None
    for root in roots:
        if np.isreal(root) and np.isfinite(root):
            m_cand = root.real
            if M_ratio[0] <= m_cand <= M_ratio[-1]:
                # Second derivative > 0 confirms local minimum
                second_deriv = 2*c + 6*d * m_cand
                if second_deriv > 0:
                    m_min = m_cand
                    break
    
    if m_min is None:
        return np.nan, np.nan
    
    # Simplified error estimate from parameter uncertainties
    param_err = np.sqrt(np.diag(pcov))
    delta_m_min = np.mean(param_err[1:]) * 0.5  # Errors on b,c,d
    
    return m_min, delta_m_min

# %%
def robust_gaussian_fit(x_data, y_data, verbose=False):
    """
    Fits Gaussian with multiple fallbacks. Returns (A, mu, sigma) or None.
    """
    def gaussian(x, A, mu, sigma):
        return A * np.exp(-0.5 * (x - mu)**2 / sigma**2)
    
    # Step 1: Basic normalization
    baseline = np.median(y_data)  # More robust than min
    peak_idx = np.argmax(y_data)
    peak_height = y_data[peak_idx] - baseline
    if peak_height <= 0:
        return None
    
    y_norm = (y_data - baseline) / peak_height
    
    # Step 2: Restrict to peak region (±3σ guess)
    x_range = x_data[-1] - x_data[0]
    peak_region = (x_data > x_data[peak_idx] - 0.2*x_range) & \
                  (x_data < x_data[peak_idx] + 0.2*x_range)
    
    x_fit, y_fit = x_data[peak_region], y_norm[peak_region]
    
    # Step 3: Try fit #1 - with good p0 + bounds
    p0 = [1.0, x_data[peak_idx], x_range*0.08]  # sigma ~8% of range
    bounds = ([0, x_fit[0], 0.5], [2, x_fit[-1], x_range*0.25])
    
    try:
        popt, _ = curve_fit(gaussian, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=5000)
        if popt[2] > 0:  # Sanity check
            return popt[0] * peak_height, popt[1], popt[2]
    except:
        pass
    
    # Step 4: Fallback #1 - Even tighter bounds
    bounds_tight = ([0, x_fit[int(len(x_fit)*0.3)], 1], 
                    [2, x_fit[int(len(x_fit)*0.7)], 20])
    try:
        popt, _ = curve_fit(gaussian, x_fit, y_fit, p0=p0, 
                           bounds=bounds_tight, maxfev=5000)
        if popt[2] > 0:
            return popt[0] * peak_height, popt[1], popt[2]
    except:
        pass
    
    # Step 5: Fallback #2 - Simple FWHM estimate (no curve_fit)
    half_max = 0.5 * np.max(y_fit)
    above_half = x_fit[y_fit > half_max]
    if len(above_half) > 2:
        fwhm = above_half[-1] - above_half[0]
        sigma_est = fwhm / (2 * np.sqrt(2 * np.log(2)))  # FWHM → σ conversion
        return peak_height, x_data[peak_idx], sigma_est
    
    if verbose:
        print("All fitting attempts failed")
    return None

def _plt(refVec, z, mr, path, ind):
    fig = plt.figure()
    plt.imshow(refVec, cmap='gray')
    # plt.title("Refocused image of magnification ratio = "+ str(round(mr, 3)) )
    plt.title("Refocused image at "+ str(round(z, 3)) + ' mm' )
    plt.colorbar()
    fig.savefig(join(path, str(ind + 1) + f"_refocused_z_{round(z, 3)}_mr_{mr}.png"), dpi='figure',transparent=False)
    plt.close("all")

def plt_sigma(M_ratio, sigma, z, path, cyc, axis='x'):
    save_path = join(path, 'sigma_plots', axis)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig = plt.figure()
    # plt.scatter(M_ratio, sigma, marker='o')
    # plt.xlabel("Magnification Ratio")
    plt.scatter(z, sigma, marker='o')
    plt.xlabel("Refocused position (mm)")
    plt.ylabel("Sigma")
    # plt.title("Width of Each Ratio")
    plt.title("Width of Each position")
    # fig.savefig(join(save_path, f"{cyc}_sigma_z_{round(z)}.png"), dpi='figure',transparent=False)
    fig.savefig(join(save_path, f"{cyc}_sigma_z_{round(M_ratio, 3)}.png"), dpi='figure',transparent=False)
    plt.close("all")

def refocusing(G2, z, mr, rangeA, rangeB, dpA, dpB, NA, NB, path, ind, maxInt=None):
    # cpi.Print("Refocusing","{} of {}".format(counter+1,len(M_ratio)))
    # cpi.Print("Refocusing","{} of {}".format(counter+1,len(REFOC)))
    matrix  = transf(z, mr)
    invMat  = np.linalg.inv(matrix)
    dRef    = np.sqrt((invMat[0,0]*dpA)**2 + (invMat[0,1]*dpB)**2)
    maxRef  = (np.abs(invMat[0,0])*dpA*NA + np.abs(invMat[0,1])*dpB*NB)/2
    rangeRef= np.arange(-maxRef, maxRef, dRef)
    NR      = len(rangeRef)
    dSum    = np.sqrt((matrix[0,1]/dpA)**2 + (matrix[1,1]/dpB)**2)**-1
    maxSum  = maxInt if maxInt else (np.abs(invMat[1,0])*dpA*NA + np.abs(invMat[1,1])*dpB*NB)/2
    rangeSum= np.arange(-maxSum, maxSum, dSum)
    # NS      = len(rangeSum)
    points  = (rangeA, rangeA, rangeB, rangeB)
    refPts  = np.array([x for x in itertools.product(rangeRef, rangeRef)])
    sumPts  = np.array([x for x in itertools.product(rangeSum, rangeSum)])
    # newPixels = [x for x in range(len(rangeRef)**2)]
    matrix4D  = np.array(
        [[matrix[0,0],0,matrix[0,1],0],
         [0,matrix[0,0],0,-matrix[0,1]],
         [matrix[1,0],0,matrix[1,1],0],
         [0,matrix[1,0],0,matrix[1,1]]], dtype=np.float64
        )
    method='nearest'
    # matrix4D = np.linalg.inv(matrix4D)
    # newPts  = np.array([[[i,j,k,l] for k in rangeSum for l in rangeSum] for i in rangeRef for j in rangeRef])
    def RefocusSinglePixel(pixelCoords):
        newPoints = np.insert(sumPts,[0,0],refPts[pixelCoords],axis=1)
        newPoints = np.matmul(matrix4D, newPoints.T).T
        outcome   = np.sum(interp(points, G2, newPoints, method=method,
                             bounds_error=False,fill_value=0))
        return outcome
    # pool   = mp.Pool(8)
    refVec = list(map(RefocusSinglePixel, range(len(rangeRef)**2)))
    refVec=np.array(refVec).reshape((NR,NR))

    _plt(refVec, z, mr, path, ind)
    return refVec


class Calculating_G2():
    def __init__(self, fileA, fileB, frames=None):
        self.fileA = imread(fileA).astype("float64")
        self.fileB = imread(fileB).astype("float64")
        if frames is not None:
            self.fileA = self.fileA[:frames].astype("float64")
            self.fileB = self.fileB[:frames].astype("float64")

    def _bin_3d_array(self, arr, bin_size):
        """
        Bins a 3D NumPy array along the last two axes while preserving the first axis.

        Parameters:
        arr (numpy.ndarray): Input 3D array of shape (N, H, W).
        bin_size (int): The factor by which to downsample the last two dimensions.

        Returns:
        numpy.ndarray: Binned array of shape (N, H//bin_size, W//bin_size).
        """
        if arr.ndim != 3:
            raise ValueError("Input array must be 3D.")
    
        N, H, W = arr.shape
        if H % bin_size != 0 or W % bin_size != 0:
            arr = arr[:, H % bin_size:, W % bin_size:]

        # Reshape and sum over the new axes
        arr_binned = arr.reshape(N, H//bin_size, bin_size, W//bin_size, bin_size).sum(axis=(2, 4))
    
        return arr_binned
    
    def correlation(self, binA, binB):
        N = self.fileA.shape[0]
        # NA = self.fileA.shape[1]//binA
        # NB = self.fileB.shape[1]//binB
        NA1 = self.fileA.shape[1]//binA
        NB1 = self.fileB.shape[1]//binB
        NA2 = self.fileA.shape[2]//binA
        NB2 = self.fileB.shape[2]//binB
        spatial = self._bin_3d_array(self.fileA, binA).reshape(N, NA1*NA2).astype("float64")
        angular = self._bin_3d_array(self.fileB, binB).reshape(N, NB1*NB2).astype("float64")

        # self.G2 = np.matmul(spatial.T, angular) / np.tensordot(np.sum(spatial,0), np.mean(angular,0), axes=0) - 1
        self.G2 = np.matmul(spatial.T, angular)/N - np.tensordot(np.mean(spatial,0), np.mean(angular,0), axes=0)
        self.G2 = self.G2.reshape(NA1, NA2, NB1, NB2)
        return self.G2
    
    def padding(self, pad):
        self.G2 = np.pad(self.G2, ((pad, pad), (pad, pad), (0,0), (0,0)))
        return self.G2

class Timer:
    def __init__(self):
            self.start_times = {}
            self.elapsed_times = {}
    def start(self, label):
        #   self.start_times[label] = time.time()
          self.start_times[label] = time.perf_counter()
    def stop(self, label):
        if label in self.start_times:
            # elapsed_time = time.time() - self.start_times[label]
            elapsed_time = time.perf_counter() - self.start_times[label]
            self.elapsed_times[label] = elapsed_time
        else:
             print(f"Timer for {label} is not started.")
    def savefile(self, filename):
        with open(filename, 'w') as file:
            for label in self.elapsed_times:
                file.write(f"{label}, {self.elapsed_times[label]}\n")
