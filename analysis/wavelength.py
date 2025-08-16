import numpy as np
import pandas as pd


def fft_wavelength(bin_centers, avg_length):
    r = bin_centers.values
    L = avg_length.values

    # --- Detrend (quadratic) ---
    coeffs = np.polyfit(r, L, 2)
    trend = np.polyval(coeffs, r)
    L_detrended = L - trend

    # --- Interpolate to uniform grid ---
    r_uniform = np.linspace(avg_length.index[0].left,
                            avg_length.index[-1].right, len(r))
    L_uniform = np.interp(r_uniform, r, L_detrended)
    L_uniform -= L_uniform.mean()

    # --- FFT ---
    fft = np.fft.rfft(L_uniform)
    freqs = np.fft.rfftfreq(len(r_uniform), d=(r_uniform[1] - r_uniform[0]))
    power = np.abs(fft)**2

    idx = np.argmax(power[1:]) + 1 if len(power) > 1 else 0
    dominant_freq = freqs[idx]
    wavelength_fft = 1.0 / dominant_freq if dominant_freq != 0 else np.nan

    return wavelength_fft


def find_wavelength_from_data(particles: pd.DataFrame):
    # bin particles by their dist_center
    particles["dist_center"] = np.sqrt(
        particles["x"]**2 + particles["y"]**2 + particles["z"]**2)

    # bin particles by their dist_center
    particles["bin"] = pd.cut(particles["dist_center"], np.arange(
        0, particles["dist_center"].max() + 1, 1))

    # compute average length per bin
    avg_length = particles.groupby("bin")["lengths_x"].mean()

    # get bin centers for plotting
    bin_centers = avg_length.index.map(lambda interval: interval.mid)

    return bin_centers, avg_length, fft_wavelength(bin_centers, avg_length)
