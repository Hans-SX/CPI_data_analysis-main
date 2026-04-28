#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 17:04:17 2026

@author: massaro
"""
#%%
import numpy as np
import matplotlib.pyplot as plt

class LineEstimator2D:
    def __init__(self, I, x=None, y=None):
        """
        I : 2D numpy array (intensity map)
        x, y : optional coordinate arrays (same shape as I or 1D axes)
        """
        self.I = np.asarray(I, dtype=float)

        ny, nx = self.I.shape

        # Build coordinate grids
        if x is None or y is None:
            x = np.arange(nx) - nx // 2
            y = np.arange(ny) - ny // 2
            self.X, self.Y = np.meshgrid(x, y)
        else:
            if x.ndim == 1 and y.ndim == 1:
                self.X, self.Y = np.meshgrid(x, y)
            else:
                self.X, self.Y = x, y

        # Normalize safely
        self.weights = self.I.copy()
        total = np.sum(self.weights)
        if total <= 0:
            raise ValueError("Intensity sum must be positive")
        self.weights /= total

    def compute_centroid(self):
        w = self.weights
        X, Y = self.X, self.Y

        self.x0 = np.sum(w * X)
        self.y0 = np.sum(w * Y)

        return self.x0, self.y0

    def compute_covariance(self):
        if not hasattr(self, "x0"):
            self.compute_centroid()

        w = self.weights
        Xc = self.X - self.x0
        Yc = self.Y - self.y0

        Sxx = np.sum(w * Xc * Xc)
        Syy = np.sum(w * Yc * Yc)
        Sxy = np.sum(w * Xc * Yc)

        self.Sigma = np.array([[Sxx, Sxy],
                               [Sxy, Syy]])
        return self.Sigma

    def estimate_direction(self):
        """
        Returns normalized (alpha, beta)
        corresponding to direction of strongest compression
        """
        if not hasattr(self, "Sigma"):
            self.compute_covariance()

        eigvals, eigvecs = np.linalg.eigh(self.Sigma)

        # smallest eigenvalue → direction of (alpha, beta)
        idx = np.argmin(eigvals)
        v = eigvecs[:, idx]

        # normalize
        v = v / np.linalg.norm(v)

        self.alpha, self.beta = v
        self.eigvals = eigvals

        return self.alpha, self.beta

    def estimate_angle(self):
        """
        Returns angle θ such that:
        alpha = cosθ, beta = sinθ
        """
        if not hasattr(self, "alpha"):
            self.estimate_direction()

        return np.arctan2(self.beta, self.alpha)

    def anisotropy_ratio(self):
        """
        Useful diagnostic: ratio of eigenvalues
        """
        if not hasattr(self, "eigvals"):
            self.estimate_direction()

        return self.eigvals.max() / self.eigvals.min()

    def bootstrap_uncertainty(self, n_samples=100):
        """
        Rough uncertainty estimate via bootstrap resampling
        """
        ny, nx = self.I.shape
        flat_idx = np.arange(nx * ny)

        probs = self.weights.flatten()

        angles = []

        for _ in range(n_samples):
            # sample pixels with replacement
            idx = np.random.choice(flat_idx, size=flat_idx.size, p=probs)

            w_boot = np.zeros_like(probs)
            np.add.at(w_boot, idx, 1)
            w_boot = w_boot.reshape(self.I.shape)
            w_boot = w_boot / np.sum(w_boot)

            Xc = self.X - self.x0
            Yc = self.Y - self.y0

            Sxx = np.sum(w_boot * Xc * Xc)
            Syy = np.sum(w_boot * Yc * Yc)
            Sxy = np.sum(w_boot * Xc * Yc)

            Sigma = np.array([[Sxx, Sxy],
                              [Sxy, Syy]])

            eigvals, eigvecs = np.linalg.eigh(Sigma)
            v = eigvecs[:, np.argmin(eigvals)]
            angle = np.arctan2(v[1], v[0])

            angles.append(angle)

        angles = np.unwrap(angles)
        return np.std(angles)
   


#%%
def main():
    # synthetic test
    nx, ny = 100, 100
    x = np.linspace(-10, 10, nx)
    y = np.linspace(-10, 10, ny)
    X, Y = np.meshgrid(x, y)

    alpha_true = 1
    beta_true = .1

    width  = 1
    range_ = 5
    noise  = 0

    I = np.exp(-X**2 / (2 * range_**2)) \
        * np.exp(-(alpha_true * X + beta_true * Y)**2 / (2 * width**2))
    I = I + noise * np.random.rand(nx,ny)
   
    # I[I<I.max()/5]=0

    plt.imshow(I)
    plt.show()
    estimator = LineEstimator2D(I, x, y)

    alpha, beta = estimator.estimate_direction()
    theta = estimator.estimate_angle()
    ratio = estimator.anisotropy_ratio()
    err = estimator.bootstrap_uncertainty(50)

    print("Estimated (alpha, beta):", alpha, beta)
    print("True (alpha, beta):     ", alpha_true, beta_true)
    print("Angle (deg):", theta/np.pi*180+180)
    print("Anisotropy ratio:", ratio)
    print("Angle uncertainty ~", err)

#%%
if __name__ == "__main__":
    main()