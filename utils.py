import matplotlib.pyplot as plt
import os
import time
from tifffile import imread
import itertools
from os.path import join

import numpy as np
from scipy.optimize  import curve_fit, least_squares
from scipy.interpolate import interpn as interp
from scipy.ndimage import gaussian_filter
from odrpack import odr_fit


from config import transf

def ridge_fit(
    g2x,
    sigma_smooth=0,
    half_window=10,
    max_jump=4,
    intensity_threshold=0.20,
    fit_radius=30,
    intensity_power=5,
    bootstrapping=False
):
    """
    For fit_every_ang_max.py.
    Ridge tracking + weighted linear fit.

    Parameters
    ----------
    g2x : 2D numpy array
        Input image.

    sigma_smooth : float
        Gaussian smoothing sigma.

    half_window : int
        Vertical search window around previous point.

    max_jump : float
        Maximum allowed vertical jump between columns.

    intensity_threshold : float
        Keep only points with intensity above
        threshold * max_intensity.

    fit_radius : int
        Only fit points within this horizontal
        distance from the brightest point.

    intensity_power : float
        Weight exponent:
            w = intensity ** intensity_power

    plot_result : bool
        Display result.

    bootstrapping : bool
        Whether to perform bootstrapping.

    Returns
    -------
    result : dict
    """

    # ============================================================
    # Smooth image
    # ============================================================

    Is = gaussian_filter(g2x, sigma=sigma_smooth)

    Ny, Nx = Is.shape

    # ============================================================
    # Brightest point
    # ============================================================

    y0, x0 = np.unravel_index(np.argmax(Is), Is.shape)

    # ============================================================
    # Subpixel quadratic refinement
    # ============================================================

    def refine_peak(profile, idx):

        if idx <= 0 or idx >= len(profile)-1:
            return float(idx)

        y1 = profile[idx-1]
        y2 = profile[idx]
        y3 = profile[idx+1]

        denom = y1 - 2*y2 + y3

        if abs(denom) < 1e-12:
            return float(idx)

        delta = 0.5 * (y1 - y3) / denom

        return idx + delta

    # ============================================================
    # Ridge tracking
    # ============================================================

    def follow(step):

        xs = []
        ys = []
        intensities = []

        x = x0
        y_prev = float(y0)

        while True:

            x += step

            if x < 0 or x >= Nx:
                break

            ymin = max(0, int(round(y_prev - half_window)))
            ymax = min(Ny, int(round(y_prev + half_window + 1)))

            profile = Is[ymin:ymax, x]

            if len(profile) < 3:
                break

            # ----------------------------------------------------
            # Local maximum
            # ----------------------------------------------------

            idx_local = np.argmax(profile)

            # ----------------------------------------------------
            # Subpixel refinement
            # ----------------------------------------------------

            idx_refined = refine_peak(profile, idx_local)

            yc = ymin + idx_refined

            # ----------------------------------------------------
            # Continuity constraint
            # ----------------------------------------------------

            if abs(yc - y_prev) > max_jump:
                yc = y_prev

            # ----------------------------------------------------
            # Intensity at ridge point
            # ----------------------------------------------------

            intensity = np.interp(
                yc,
                np.arange(Ny),
                Is[:, x]
            )

            xs.append(x)
            ys.append(yc)
            intensities.append(intensity)

            y_prev = yc

        return xs, ys, intensities

    # ============================================================
    # Track both directions
    # ============================================================

    xs_r, ys_r, Is_r = follow(+1)
    xs_l, ys_l, Is_l = follow(-1)

    # xs_l, ys_l, Is_l are lists, + will concatenate them. We want to reverse the left side to have increasing x order.
    ridge_x = np.array(xs_l[::-1] + [x0] + xs_r)
    ridge_y = np.array(ys_l[::-1] + [y0] + ys_r)

    ridge_I = np.array(Is_l[::-1] + [Is[y0, x0]] + Is_r)

    # ============================================================
    # Point selection
    # ============================================================

    mask_intensity = (
        ridge_I >
        intensity_threshold * ridge_I.max()
    )

    mask_radius = (
        np.abs(ridge_x - x0) < fit_radius
    )

    mask = mask_intensity & mask_radius

    xfit = ridge_x[mask]
    yfit = ridge_y[mask]
    Ifit = ridge_I[mask]

    # ============================================================
    # Weighted linear fit
    # ============================================================
    
    weights = Ifit**intensity_power

    coeffs, cov = np.polyfit(
        xfit,
        yfit,
        deg=1,
        w=np.sqrt(weights),
        cov=True
    )
    
    slope = coeffs[0]
    intercept = coeffs[1]
    
    # ============================================================
    # Residuals
    # ============================================================
    
    y_model = slope * xfit + intercept
    
    residuals = yfit - y_model
    
    N = len(xfit)
    
    # ============================================================
    # Weighted residual variance
    # ============================================================
    
    s2 = np.sum(weights * residuals**2) / (N - 2)
    
    # ============================================================
    # Weighted x mean
    # ============================================================
    
    xw_mean = np.sum(weights * xfit) / np.sum(weights)
    
    # ============================================================
    # Slope uncertainty
    # ============================================================
    
    Sxx = np.sum(weights * (xfit - xw_mean)**2)
    
    slope_err = np.sqrt(s2 / Sxx)
    
    # ============================================================
    # Angle + uncertainty
    # ============================================================
    
    angle_rad = np.arctan(slope)
    
    angle_deg = np.degrees(angle_rad)
    
    angle_err_rad = slope_err / (1 + slope**2)
    
    angle_err_deg = np.degrees(angle_err_rad)
    
    # print(
    #     f"Angle = {angle_deg:.2f} ± "
    #     f"{angle_err_deg:.2f} deg"
    # )

    # ============================================================
    # Return
    # ============================================================
    if bootstrapping != False:
        slope_bt = []
        intercept_bt = []
        while bootstrapping > 0:
            rng = np.random.default_rng()
            ridx = rng.choice(len(xfit), size=len(xfit), replace=True)
            ridx = np.unique(ridx)

            xfit_bt = xfit[ridx]
            yfit_bt = yfit[ridx]
            Ifit_bt = Ifit[ridx]
            rng = np.random.default_rng()
            ridx = rng.choice(len(xfit), size=len(xfit), replace=True)
            ridx = np.unique(ridx)

            xfit_bt = xfit[ridx]
            yfit_bt = yfit[ridx]
            Ifit_bt = Ifit[ridx]

            weights_bt = Ifit_bt**intensity_power

            coeffs = np.polyfit(
            xfit_bt,
            yfit_bt,
            deg=1,
            w=np.sqrt(weights_bt)
            )
            slope_bt.append(coeffs[0])
            intercept_bt.append(coeffs[1])
            bootstrapping -= 1

        slope_bt_mean = np.mean(np.array(slope_bt))
        slope_bt_err = np.std(np.array(slope_bt))
        return {
            "slope": slope,
            "slope_err": slope_err,
            "slope_bt_mean": slope_bt_mean,
            "slope_bt_std": slope_bt_err,
            "intercept_bt_mean": np.mean(np.array(intercept_bt)),
            "x_fit": xfit_bt,
            "y_fit": yfit_bt
        }

    return {
        "x_all": ridge_x,
        "y_all": ridge_y,
        "I_all": ridge_I,
        "x_fit": xfit,
        "y_fit": yfit,
        "weights": weights,
        "slope": slope,
        "intercept": intercept,
        "angle_deg": angle_deg,
        "slope_err": slope_err,
        "cov": cov
    }

def ridge_odr_fit(
    g2x,
    sigma_smooth=0,
    half_window=10,
    max_jump=4,
    intensity_threshold=0.20,
    fit_radius=30,
    intensity_power=5,
    bootstrapping=False
):
    """
    For fit_every_ang_max.py.
    Ridge tracking + weighted linear odr fit.

    Parameters
    ----------
    g2x : 2D numpy array
        Input image.

    sigma_smooth : float
        Gaussian smoothing sigma.

    half_window : int
        Vertical search window around previous point.

    max_jump : float
        Maximum allowed vertical jump between columns.

    intensity_threshold : float
        Keep only points with intensity above
        threshold * max_intensity.

    fit_radius : int
        Only fit points within this horizontal
        distance from the brightest point.

    intensity_power : float
        Weight exponent:
            w = intensity ** intensity_power

    plot_result : bool
        Display result.

    bootstrapping : bool
        Whether to perform bootstrapping.

    Returns
    -------
    result : dict
    """

    # ============================================================
    # Smooth image
    # ============================================================

    Is = gaussian_filter(g2x, sigma=sigma_smooth)

    Ny, Nx = Is.shape

    # ============================================================
    # Brightest point
    # ============================================================

    y0, x0 = np.unravel_index(np.argmax(Is), Is.shape)

    # ============================================================
    # Subpixel quadratic refinement
    # ============================================================

    def refine_peak(profile, idx):

        if idx <= 0 or idx >= len(profile)-1:
            return float(idx)

        y1 = profile[idx-1]
        y2 = profile[idx]
        y3 = profile[idx+1]

        denom = y1 - 2*y2 + y3

        if abs(denom) < 1e-12:
            return float(idx)

        delta = 0.5 * (y1 - y3) / denom

        return idx + delta

    # ============================================================
    # Ridge tracking
    # ============================================================

    def follow(step):

        xs = []
        ys = []
        intensities = []

        x = x0
        y_prev = float(y0)

        while True:

            x += step

            if x < 0 or x >= Nx:
                break

            ymin = max(0, int(round(y_prev - half_window)))
            ymax = min(Ny, int(round(y_prev + half_window + 1)))

            profile = Is[ymin:ymax, x]

            if len(profile) < 3:
                break

            # ----------------------------------------------------
            # Local maximum
            # ----------------------------------------------------

            idx_local = np.argmax(profile)

            # ----------------------------------------------------
            # Subpixel refinement
            # ----------------------------------------------------

            idx_refined = refine_peak(profile, idx_local)

            yc = ymin + idx_refined

            # ----------------------------------------------------
            # Continuity constraint
            # ----------------------------------------------------

            if abs(yc - y_prev) > max_jump:
                yc = y_prev

            # ----------------------------------------------------
            # Intensity at ridge point
            # ----------------------------------------------------

            intensity = np.interp(
                yc,
                np.arange(Ny),
                Is[:, x]
            )

            xs.append(x)
            ys.append(yc)
            intensities.append(intensity)

            y_prev = yc

        return xs, ys, intensities

    # ============================================================
    # Track both directions
    # ============================================================

    xs_r, ys_r, Is_r = follow(+1)
    xs_l, ys_l, Is_l = follow(-1)

    # xs_l, ys_l, Is_l are lists, + will concatenate them. We want to reverse the left side to have increasing x order.
    ridge_x = np.array(xs_l[::-1] + [x0] + xs_r)
    ridge_y = np.array(ys_l[::-1] + [y0] + ys_r)

    ridge_I = np.array(Is_l[::-1] + [Is[y0, x0]] + Is_r)

    # ============================================================
    # Point selection
    # ============================================================

    mask_intensity = (
        ridge_I >
        intensity_threshold * ridge_I.max()
    )

    mask_radius = (
        np.abs(ridge_x - x0) < fit_radius
    )

    mask = mask_intensity & mask_radius

    xfit = ridge_x[mask]
    yfit = ridge_y[mask]
    Ifit = ridge_I[mask]

    # ============================================================
    # Weighted linear fit
    # ============================================================
    def line_for_odr(x: np.narray, beta: np.ndarray) -> np.ndarray:
        b1, b2 = beta
        return b1*x + b2
    weights = Ifit**intensity_power

    sol = np.odr_fit(
        line_for_odr,
        xfit,
        yfit,
        beta0 = [1.0, 1.0]
        )
    
    slope = sol.beta[0]
    intercept = sol.beta[1]
    slope_err = sol.sd_beta[0]
    """
    # ============================================================
    # Residuals
    # ============================================================
    
    y_model = slope * xfit + intercept
    
    residuals = yfit - y_model
    
    N = len(xfit)
    
    # ============================================================
    # Weighted residual variance
    # ============================================================
    
    s2 = np.sum(weights * residuals**2) / (N - 2)
    
    # ============================================================
    # Weighted x mean
    # ============================================================
    
    xw_mean = np.sum(weights * xfit) / np.sum(weights)
    
    # ============================================================
    # Slope uncertainty
    # ============================================================
    
    Sxx = np.sum(weights * (xfit - xw_mean)**2)
    
    slope_err = np.sqrt(s2 / Sxx)
    """
    # ============================================================
    # Angle + uncertainty
    # ============================================================
    
    angle_rad = np.arctan(slope)
    
    angle_deg = np.degrees(angle_rad)
    
    angle_err_rad = slope_err / (1 + slope**2)
    
    angle_err_deg = np.degrees(angle_err_rad)
    
    # print(
    #     f"Angle = {angle_deg:.2f} ± "
    #     f"{angle_err_deg:.2f} deg"
    # )

    # ============================================================
    # Return
    # ============================================================
    return {
        "x_all": ridge_x,
        "y_all": ridge_y,
        "I_all": ridge_I,
        "x_fit": xfit,
        "y_fit": yfit,
        "weights": weights,
        "slope": slope,
        "intercept": intercept,
        "angle_deg": angle_deg,
        "slope_err": slope_err,
    }

def plot_ridge_fit(result, g2, sigma_smooth=0, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    Is = gaussian_filter(g2, sigma=sigma_smooth)
    Ny, Nx = Is.shape
    y0, x0 = np.unravel_index(np.argmax(Is), Is.shape)

    xx = np.arange(Nx)
    yy = result['slope'] * xx + result['intercept']

    ax.imshow(Is, origin='upper')

    # all tracked points
    ax.plot(
        result['x_all'],
        result['y_all'],
        'r.',
        ms=5,
        alpha=0.4
    )

    # fitted points
    ax.plot(
        result['x_fit'],
        result['y_fit'],
        'wo',
        ms=4
    )

    # fitted line
    ax.plot(
        xx,
        yy,
        'w--',
        lw=3
    )

    ax.scatter(
        [x0],
        [y0],
        c='cyan',
        s=80
    )

    ax.set_title(
        f"Slope = {result['slope']:.4f}    "
        f"Angle = {result['angle_deg']:.2f} deg"
    )

    ax.set_xlim(0, Nx)
    ax.set_ylim(Ny, 0)

    return ax


def column_centroids(image):
    """
    Did not work as a charm.
    """
    image = np.asarray(image, dtype=float)
    if image.ndim == 3:
        image = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]

    h, w = image.shape
    y = np.arange(h)

    col_sums = image.sum(axis=0)
    weighted_sums = (image * y[:, None]).sum(axis=0)

    centroids = np.full(w, np.nan)
    valid = col_sums > 0
    centroids[valid] = weighted_sums[valid] / col_sums[valid]

    # print("Column sums (first 10):", col_sums[:10])
    # print("Centroids (first 10):", centroids[:10])

    # if ax is None:
    #     fig, ax = plt.subplots()

    # ax.imshow(image, cmap='gray', origin='upper')
    # x = np.arange(w)
    # ax.scatter(x[valid], centroids[valid], c=color, s=s)
    # ax.set_xlim(-0.5, w - 0.5)
    # ax.set_ylim(h - 0.5, -0.5)
    # ax.set_title("Column centroids (x vs y)")

    return centroids

def line(x, m, b):
    return m*x + b

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

def fit_gaussian_2D(data, ls=False):
    def coordinate(shape):
        y, x = np.indices(shape)
        y = y - shape[0] // 2
        x = x - shape[1] // 2
        coords = np.vstack((x.ravel(), y.ravel()))
        return coords

    def Gaussian2D(coords, A, x0, y0, sigma_x, offset=0):
        x, y = coords
        return A * np.exp(-(((x - x0) ** 2  + (y - y0) ** 2)/ (2 * sigma_x ** 2))) + offset
    
    def init_guess(data):
        y, x = np.indices(data.shape)
        y = y - data.shape[0] // 2
        x = x - data.shape[1] // 2
        Amp_guess = data.max() - data.min()
        x0_guess = x[np.where(data==data.max())][0]
        y0_guess = y[np.where(data==data.max())][0]
        sigma_x_guess = 5  # Approximate width
        offset_guess = data.min()
        p0 = [Amp_guess, x0_guess, y0_guess, sigma_x_guess, offset_guess]
        return p0
    
    def residuals(params, data):
        A, x0, y0, sigma_x, offset = params
        model = Gaussian2D(coordinate(data.shape), A, x0, y0, sigma_x, offset).reshape(data.shape)
        return (model - data).ravel()
    
    p0 = init_guess(data)
    if ls:
        result = least_squares(residuals, p0, args=(data,), loss='linear', max_nfev=5000)
        popt = result.x
        # pcov is not directly available from least_squares, so we can estimate it from the Jacobian at the solution
        J = result.jac
        residual_variance = np.sum(result.fun**2) / (len(data.ravel()) - len(popt))
        # pcov = np.linalg.inv(J.T @ J) * residual_variance
        pcov = np.arange(25).reshape(5, 5) * residual_variance

    else:
        popt, pcov = curve_fit(Gaussian2D, coordinate(data.shape), data.ravel(), p0=p0, maxfev=5000)

    Amp, x0, y0, sigma_x, offset = popt
    return Amp, x0, y0, sigma_x, offset, np.diag(pcov)[3], p0[1:3]

# %%
def robust_gaussian_fit(x_data, y_data):
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
        print("No valid peak found")
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
        popt, pcov = curve_fit(gaussian, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=5000)
        if popt[2] > 0:  # Sanity check
            return popt[0] * peak_height, popt[1], popt[2], np.diag(pcov)[2]
        else:
            print("Unphysical sigma from fit #1")
    except:
        print("Error occurred while fitting Gaussian #1")
        pass
    
    # Step 4: Fallback #1 - Even tighter bounds
    bounds_tight = ([0, x_fit[int(len(x_fit)*0.3)], 1], 
                    [2, x_fit[int(len(x_fit)*0.7)], 20])
    try:
        popt, pcov = curve_fit(gaussian, x_fit, y_fit, p0=p0, 
                           bounds=bounds_tight, maxfev=5000)
        if popt[2] > 0:
            return popt[0] * peak_height, popt[1], popt[2], np.diag(pcov)[2]
        else:
            print("Unphysical sigma from fit #2")
    except:
        print("Error occurred while fitting Gaussian #2")
        pass
    
    # Step 5: Fallback #2 - Simple FWHM estimate (no curve_fit)
    half_max = 0.5 * np.max(y_fit)
    above_half = x_fit[y_fit > half_max]
    if len(above_half) > 2:
        fwhm = above_half[-1] - above_half[0]
        sigma_est = fwhm / (2 * np.sqrt(2 * np.log(2)))  # FWHM → σ conversion
        return peak_height, x_data[peak_idx], sigma_est, np.nan
    

    print("Warning: Gaussian fit failed, returning None")
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

def refocusing(G2, z, mr, rangeA, rangeB, dpA, dpB, NA, NB, path, ind=0, maxInt=None):
    # cpi.Print("Refocusing","{} of {}".format(counter+1,len(M_ratio)))
    # cpi.Print("Refocusing","{} of {}".format(counter+1,len(REFOC)))
    matrix  = transf(z, mr)
    invMat  = np.linalg.inv(matrix)
    # Ref: alpha xa + beta xb = xr
    dRef    = np.sqrt((invMat[0,0]*dpA)**2 + (invMat[0,1]*dpB)**2)
    maxRef  = (np.abs(invMat[0,0])*dpA*NA + np.abs(invMat[0,1])*dpB*NB)/2
    # rangeRef, new grid in refocused space
    rangeRef= np.arange(-maxRef, maxRef, dRef)
    NR      = len(rangeRef)
    dSum    = np.sqrt((matrix[0,1]/dpA)**2 + (matrix[1,1]/dpB)**2)**-1
    maxSum  = maxInt if maxInt else (np.abs(invMat[1,0])*dpA*NA + np.abs(invMat[1,1])*dpB*NB)/2
    rangeSum= np.arange(-maxSum, maxSum, dSum)
    # NS      = len(rangeSum)
    #* fixed 3 axes, check the intensity drop for 4 axes? -> slices of G2?
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
    # restrict the range of rangeRef, there should be a transformation from points to rangeRef
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
