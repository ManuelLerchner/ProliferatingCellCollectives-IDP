import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.optimize import curve_fit


def fft_wavelength_improved(bin_centers, avg_length, plot=True):
    """Improved FFT-based wavelength detection with multiple peak analysis"""
    r = bin_centers.values
    L = avg_length.values

    # --- Detrend (quadratic) ---
    coeffs = np.polyfit(r, L, 2)
    trend = np.polyval(coeffs, r)
    L_detrended = L - trend

    # --- Interpolate to uniform grid ---
    r_uniform = np.linspace(r.min(), r.max(), len(r) * 2)  # Higher resolution
    L_uniform = np.interp(r_uniform, r, L_detrended)
    L_uniform -= L_uniform.mean()

    # Apply window to reduce edge effects
    window = np.hanning(len(L_uniform))
    L_windowed = L_uniform * window

    # --- FFT ---
    fft = np.fft.rfft(L_windowed)
    freqs = np.fft.rfftfreq(len(r_uniform), d=(r_uniform[1] - r_uniform[0]))
    power = np.abs(fft)**2

    # Find peaks in power spectrum (excluding DC component)
    peaks, properties = signal.find_peaks(
        power[1:], height=np.max(power[1:]) * 0.1)
    peaks += 1  # Adjust for excluding DC component

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot detrended data
        ax1.plot(r, L_detrended, 'b-', linewidth=2)
        ax1.set_xlabel('Distance')
        ax1.set_ylabel('Detrended Length')
        ax1.set_title('Detrended Data')
        ax1.grid(True)

        # Plot power spectrum
        wavelengths = 1.0 / freqs[1:]
        ax2.semilogy(wavelengths, power[1:], 'b-', linewidth=2)
        ax2.set_xlabel('Wavelength')
        ax2.set_ylabel('Power')
        ax2.set_title('Power Spectrum')
        ax2.grid(True)

        # Mark peaks
        if len(peaks) > 0:
            peak_wavelengths = 1.0 / freqs[peaks]
            ax2.scatter(peak_wavelengths,
                        power[peaks], color='red', s=100, zorder=5)
            for i, (wl, p) in enumerate(zip(peak_wavelengths, power[peaks])):
                ax2.annotate(f'{wl:.1f}', (wl, p), xytext=(5, 5),
                             textcoords='offset points', fontsize=10)

        plt.tight_layout()
        plt.show()

    # Return dominant wavelength and all detected wavelengths
    if len(peaks) > 0:
        dominant_idx = peaks[np.argmax(power[peaks])]
        dominant_wavelength = 1.0 / freqs[dominant_idx]
        all_wavelengths = 1.0 / freqs[peaks]
        return dominant_wavelength, all_wavelengths, power[peaks]
    else:
        return np.nan, [], []


def autocorrelation_wavelength(bin_centers, avg_length, plot=True):
    """Find wavelength using autocorrelation method"""
    r = bin_centers.values
    L = avg_length.values

    # Detrend
    coeffs = np.polyfit(r, L, 2)
    trend = np.polyval(coeffs, r)
    L_detrended = L - trend

    # Interpolate to uniform grid
    r_uniform = np.linspace(r.min(), r.max(), len(r) * 2)
    L_uniform = np.interp(r_uniform, r, L_detrended)
    L_uniform -= L_uniform.mean()

    # Compute autocorrelation
    autocorr = np.correlate(L_uniform, L_uniform, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Take positive lags only

    # Create lag array
    dr = r_uniform[1] - r_uniform[0]
    lags = np.arange(len(autocorr)) * dr

    # Find peaks in autocorrelation (excluding zero lag)
    peaks, _ = signal.find_peaks(autocorr[1:], height=np.max(autocorr) * 0.1)
    peaks += 1  # Adjust for excluding zero lag

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(lags, autocorr, 'b-', linewidth=2)
        if len(peaks) > 0:
            plt.scatter(lags[peaks], autocorr[peaks],
                        color='red', s=100, zorder=5)
            for lag, corr in zip(lags[peaks], autocorr[peaks]):
                plt.annotate(f'{lag:.1f}', (lag, corr), xytext=(5, 5),
                             textcoords='offset points', fontsize=10)
        plt.xlabel('Lag (Distance)')
        plt.ylabel('Autocorrelation')
        plt.title('Autocorrelation Analysis')
        plt.grid(True)
        plt.show()

    # Return the first significant peak as wavelength
    if len(peaks) > 0:
        return lags[peaks[0]], lags[peaks]
    else:
        return np.nan, []


def peak_to_peak_wavelength(bin_centers, avg_length, plot=True):
    """Find wavelength by measuring distances between consecutive peaks"""
    r = bin_centers.values
    L = avg_length.values

    # Find peaks and troughs
    peaks, _ = signal.find_peaks(
        L, height=np.std(L) * 0.5, distance=5)
    troughs, _ = signal.find_peaks(-L,
                                   height=np.std(L) * 0.5, distance=5)

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(r, L, 'b-', linewidth=2, label='Detrended data')
        plt.scatter(r[peaks], L[peaks], color='red', s=100,
                    label=f'Peaks ({len(peaks)})', zorder=5)
        plt.scatter(r[troughs], L[troughs], color='orange', s=100,
                    label=f'Troughs ({len(troughs)})', zorder=5)
        plt.xlabel('Distance')
        plt.ylabel('Detrended Length')
        plt.title('Peak Detection')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Calculate wavelengths from peak-to-peak distances
    if len(peaks) > 1:
        peak_distances = np.diff(r[peaks])
        avg_wavelength = np.mean(peak_distances)
        std_wavelength = np.std(peak_distances)
        return avg_wavelength, peak_distances, std_wavelength
    else:
        return np.nan, [], np.nan


def sine_fit_wavelength(bin_centers, avg_length, plot=True):
    """Fit a sine wave to estimate wavelength"""
    r = bin_centers.values
    L = avg_length.values

    # Detrend
    coeffs = np.polyfit(r, L, 2)
    trend = np.polyval(coeffs, r)
    L_detrended = L - trend

    # Define sine function
    def sine_func(x, amplitude, wavelength, phase, offset):
        return amplitude * np.sin(2 * np.pi * x / wavelength + phase) + offset

    # Initial guess for parameters
    amplitude_guess = (np.max(L_detrended) - np.min(L_detrended)) / 2
    wavelength_guess = (r.max() - r.min()) / 3  # Assume ~3 cycles in data
    phase_guess = 0
    offset_guess = np.mean(L_detrended)

    try:
        # Fit the sine wave
        popt, pcov = curve_fit(sine_func, r, L_detrended,
                               p0=[amplitude_guess, wavelength_guess,
                                   phase_guess, offset_guess],
                               maxfev=5000)

        amplitude, wavelength, phase, offset = popt

        # Calculate R-squared
        y_pred = sine_func(r, *popt)
        ss_res = np.sum((L_detrended - y_pred) ** 2)
        ss_tot = np.sum((L_detrended - np.mean(L_detrended)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(r, L_detrended, 'bo', markersize=6, label='Data')
            r_fine = np.linspace(r.min(), r.max(), 200)
            plt.plot(r_fine, sine_func(r_fine, *popt), 'r-', linewidth=2,
                     label=f'Fitted sine (λ={wavelength:.1f}, R²={r_squared:.3f})')
            plt.xlabel('Distance')
            plt.ylabel('Detrended Length')
            plt.title('Sine Wave Fit')
            plt.legend()
            plt.grid(True)
            plt.show()

        return wavelength, r_squared, popt
    except:
        return np.nan, 0, None


def findpeaks_wavelength(bin_centers, avg_length, plot=True):
    import findpeaks
    import numpy as np

    fp = findpeaks.findpeaks(method='peakdetect', lookahead=1, interpolate=2)
    results = fp.fit(avg_length)

    df = results["df"]

    # get indices where a peak was detected
    peak_indices = df.index[df["peak"] == True].to_numpy()

    # map them to bin_centers (your x-axis)
    peak_positions = bin_centers[peak_indices]

    # distances between consecutive peaks
    peak_distances = np.diff(peak_positions)

    avg_wavelength = np.mean(peak_distances)
    std_wavelength = np.std(peak_distances)

    if plot:
        fp.plot()

    return avg_wavelength, peak_distances, std_wavelength


def comprehensive_wavelength_analysis(particles: pd.DataFrame, plot=False):
    """Run all wavelength detection methods and compare results"""
    # Prepare data (same as your original function)
    particles["dist_center"] = np.sqrt(
        particles["x"]**2 + particles["y"]**2 + particles["z"]**2)

    bins = pd.IntervalIndex.from_tuples(
        [(i, i+1) for i in range(10, int(particles["dist_center"].max()), 3)])
    particles["bin"] = pd.cut(particles["dist_center"], bins=bins)

    avg_length = particles.groupby("bin", observed=True)["lengths_x"].mean()

    bin_centers = avg_length.index.map(lambda interval: interval.mid)

    if len(avg_length) == 0:
        return None, None, None

    dominant_wl, all_wl, powers = fft_wavelength_improved(
        bin_centers, avg_length, plot=plot)

    autocorr_wl, all_autocorr_wl = autocorrelation_wavelength(
        bin_centers, avg_length, plot=plot)

    sine_wl, r_squared, params = sine_fit_wavelength(
        bin_centers, avg_length, plot=plot)

    peak_to_peak_wl, peak_distances, std_wl = peak_to_peak_wavelength(
        bin_centers, avg_length, plot=plot)

    findpeaks_wl, peak_distances, std_wl = findpeaks_wavelength(
        bin_centers, avg_length, plot=plot)

    valid_wavelengths = [w for w in [
        autocorr_wl, findpeaks_wl, peak_to_peak_wl, sine_wl] if not np.isnan(w)]

    print(valid_wavelengths)

    final_wavelength = np.mean(valid_wavelengths)

    if np.isnan(final_wavelength):
        final_wavelength = bin_centers.max()

    return bin_centers, avg_length, final_wavelength
