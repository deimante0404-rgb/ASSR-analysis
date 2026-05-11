import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

SFREQ = 2000.0
SWEEP_LEN = 16000
EPOCH_DUR = SWEEP_LEN / SFREQ
FMIN, FMAX = 20, 70
NPERSEG = 1024
NOVERLAP = 768
HOP = NPERSEG - NOVERLAP
SMOOTH_SIGMA = (0.6, 0.6)
STIM_ONSET_S = 6.0

CONDITION_DIRS = {
    "Kontrolė":           r"/Users/deimantemickute/Desktop/Bakalauras/SALINE_ASSR",
    "MK-801 1 mg/kg":     r"/Users/deimantemickute/Desktop/Bakalauras/MK1_ASSR",
    "Ketaminas 20 mg/kg": r"/Users/deimantemickute/Desktop/Bakalauras/KET20_ASSR",
}

HAS_HEADER  = True
SCALE_TO_uV = -1.0

TIME_WINDOWS = [
    ("0–30 min",  (0,  30)),
    ("60–90 min", (60, 90)),
]

def load_sweeps(folder, minute_window):
    files = sorted(glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True))
    path  = files[0]
    ep0 = int(np.ceil((minute_window[0] * 60.0) / EPOCH_DUR))
    ep1 = int(np.floor((minute_window[1] * 60.0) / EPOCH_DUR))
    nrows_needed = ep1 * SWEEP_LEN
    df = pd.read_csv(
        path, sep="\t",
        header=0 if HAS_HEADER else None,
        usecols=[1],
        dtype=np.float32,
        engine="c",
        nrows=nrows_needed
    )
    x = df.iloc[:, 0].to_numpy(copy=False)
    x = x[~np.isnan(x)]
    n_sweeps = x.size // SWEEP_LEN
    x = x[:n_sweeps * SWEEP_LEN].reshape(n_sweeps, SWEEP_LEN) * SCALE_TO_uV
    return x[ep0:min(ep1, n_sweeps)]

def compute_plf(sweeps):
    n_frames = 1 + (SWEEP_LEN - NPERSEG) // HOP
    starts   = np.arange(n_frames) * HOP
    idx      = starts[:, None] + np.arange(NPERSEG)[None, :]
    frames   = sweeps[:, idx]
    win      = np.hanning(NPERSEG).astype(np.float32)
    frames  *= win[None, None, :]
    X        = np.fft.rfft(frames, axis=2)
    X        = np.transpose(X, (0, 2, 1))
    freqs    = np.fft.rfftfreq(NPERSEG, 1 / SFREQ)
    times    = (starts + NPERSEG / 2) / SFREQ
    mask     = (freqs >= FMIN) & (freqs <= FMAX)
    X        = X[:, mask, :]
    freqs    = freqs[mask]
    eps      = 1e-12
    phase    = X / (np.abs(X) + eps)
    plf      = np.abs(np.mean(phase, axis=0))
    return freqs, times, plf

fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)

for row_idx, (win_label, win) in enumerate(TIME_WINDOWS):
    im_last = None
    for col_idx, (label, folder) in enumerate(CONDITION_DIRS.items()):
        ax = axes[row_idx, col_idx]
        sweeps = load_sweeps(folder, win)
        freqs, times, plf = compute_plf(sweeps)
        if SMOOTH_SIGMA != (0, 0):
            plf = gaussian_filter(plf, sigma=SMOOTH_SIGMA)

        onset_mask = times >= STIM_ONSET_S
        times_cut  = times[onset_mask] - STIM_ONSET_S
        plf_cut    = plf[:, onset_mask]

        extent = [times_cut[0], times_cut[-1], freqs[0], freqs[-1]]
        im_last = ax.imshow(
            plf_cut,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="magma",
            vmin=0, vmax=1,
            interpolation="nearest"
        )
        ax.grid(False)
        ax.set_title(f"{label}\n{win_label}", fontweight="bold", fontsize=10)
        ax.set_xlabel("Laikas po stimulo (s)", fontsize=9)
        if col_idx == 0:
            ax.set_ylabel("Dažnis, Hz", fontsize=9)

    fig.colorbar(im_last, ax=axes[row_idx, :], shrink=0.6,
                 label="Phase Locking Factor (PLF)")

plt.savefig(
    "/Users/deimantemickute/Desktop/Bakalauras/PLF_6_grafikai.png",
    dpi=150, bbox_inches="tight")
plt.show()