import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import shapiro, f_oneway, kruskal
from scikit_posthocs import posthoc_tukey, posthoc_dunn

SFREQ = 2000.0
SWEEP_LEN = 16000
EPOCH_DUR = SWEEP_LEN / SFREQ
FMIN, FMAX = 20, 70
NPERSEG = 1024
NOVERLAP = 768
HOP = NPERSEG - NOVERLAP
SMOOTH_SIGMA = (0.6, 0.6)
STIM_ONSET_S = 6.0

PLF_T0, PLF_T1 = 0.0, 2.0

CONDITION_DIRS = {
    "Kontrolė":           r"/Users/deimantemickute/Desktop/Bakalauras/SALINE_ASSR",
    "Ketaminas 20 mg/kg": r"/Users/deimantemickute/Desktop/Bakalauras/KET20_ASSR",
    "MK-801 1 mg/kg":     r"/Users/deimantemickute/Desktop/Bakalauras/MK1_ASSR",
}

HAS_HEADER  = True
SCALE_TO_uV = -1.0

TIME_WINDOWS = [
    ("0–30 min",  (0,  30)),
    ("60–90 min", (60, 90)),
]

GROUP_COLORS = {
    "Kontrolė":           "#7f7f7f",
    "Ketaminas 20 mg/kg": "#1f77b4",
    "MK-801 1 mg/kg":     "#d62728",
}

def load_all_animals(folder, minute_window):
    files = sorted(glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True))
    ep0 = int(np.ceil((minute_window[0] * 60.0) / EPOCH_DUR))
    ep1 = int(np.floor((minute_window[1] * 60.0) / EPOCH_DUR))
    all_sweeps = []
    for path in files:
        try:
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
            x = x[ep0:min(ep1, n_sweeps)]
            if len(x) > 0:
                all_sweeps.append(x)
                print(f"  {os.path.basename(path)}: {len(x)} epochų")
        except Exception as e:
            print(f"  Klaida {os.path.basename(path)}: {e}")
    return all_sweeps

# ================= PLF =================
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

def animal_plf_mean(sweeps, t0=PLF_T0, t1=PLF_T1):
    freqs, times, plf = compute_plf(sweeps)
    if SMOOTH_SIGMA != (0, 0):
        plf = gaussian_filter(plf, sigma=SMOOTH_SIGMA)
    times_rel = times - STIM_ONSET_S
    t_mask = (times_rel >= t0) & (times_rel <= t1)
    f_mask = (freqs >= 35) & (freqs <= 45)
    if not np.any(t_mask) or not np.any(f_mask):
        return None
    return float(plf[np.ix_(f_mask, t_mask)].mean())

def p_label(p):
    return f"p={p:.3f}"

group_labels = list(CONDITION_DIRS.keys())
tick_labels  = ["Kontrolė", "Ketaminas\n20 mg/kg", "MK-801\n1 mg/kg"]

fig, axes = plt.subplots(1, 2, figsize=(9, 6), sharey=True)

for ax, (win_label, win) in zip(axes, TIME_WINDOWS):
    print(f"\n=== {win_label} ===")

    vals = {}
    for label, folder in CONDITION_DIRS.items():
        print(f"--- {label} ---")
        all_sweeps = load_all_animals(folder, win)
        animal_vals = []
        for sweeps in all_sweeps:
            v = animal_plf_mean(sweeps)
            if v is not None:
                animal_vals.append(v)
        vals[label] = np.array(animal_vals)
        print(f"  → n={len(animal_vals)}, vidurkis={np.mean(animal_vals):.4f}")

    normal_all = True
    for g, v in vals.items():
        if len(v) >= 3:
            _, p_sh = shapiro(v)
            normal_all = normal_all and (p_sh > 0.05)

    vals_list = [vals[g] for g in group_labels]

    if normal_all:
        stat, p_main = f_oneway(*vals_list)
        test_name = "ANOVA"
        all_v = np.concatenate(vals_list)
        all_l = np.concatenate([[g]*len(vals[g]) for g in group_labels])
        df_ph = pd.DataFrame({"val": all_v, "group": all_l})
        posthoc = posthoc_tukey(df_ph, val_col="val", group_col="group")
        posthoc_name = "Tukey"
    else:
        stat, p_main = kruskal(*vals_list)
        test_name = "Kruskal-Wallis"
        all_v = np.concatenate(vals_list)
        all_l = np.concatenate([[g]*len(vals[g]) for g in group_labels])
        df_ph = pd.DataFrame({"val": all_v, "group": all_l})
        posthoc = posthoc_dunn(df_ph, val_col="val", group_col="group",
                               p_adjust="bonferroni")
        posthoc_name = "Dunn"

    print(f"\n{win_label} — {test_name}: p={p_main:.4f}")
    print(posthoc.round(4))

    positions = {g: i+1 for i, g in enumerate(group_labels)}
    rng = np.random.default_rng(42)

    for g in group_labels:
        v   = vals[g]
        pos = positions[g]
        col = GROUP_COLORS[g]

        ax.boxplot(v, positions=[pos], widths=0.45,
                   patch_artist=True,
                   medianprops=dict(color="#FFB300", lw=2.5),
                   whiskerprops=dict(color="k", lw=1.2),
                   capprops=dict(color="k", lw=1.2),
                   flierprops=dict(marker=""),
                   boxprops=dict(facecolor=col, alpha=0.5,
                                 edgecolor="k", linewidth=1.2))
        xj = rng.uniform(pos - 0.07, pos + 0.07, len(v))
        ax.scatter(xj, v, color=col, edgecolors="k",
                   s=70, zorder=5, linewidths=0.8)

    pairs = [
        ("Kontrolė", "Ketaminas 20 mg/kg"),
        ("Kontrolė", "MK-801 1 mg/kg"),
        ("Ketaminas 20 mg/kg", "MK-801 1 mg/kg"),
    ]
    all_flat = np.concatenate(vals_list)
    y_max   = np.max(all_flat)
    y_range = np.max(all_flat) - np.min(all_flat)
    bar_gap = y_range * 0.13

    for i, (g1, g2) in enumerate(pairs):
        p_val = posthoc.loc[g1, g2]
        x1, x2 = positions[g1], positions[g2]
        bar_y = y_max + bar_gap * (i + 1)
        ax.plot([x1, x1, x2, x2],
                [bar_y, bar_y + y_range*0.03,
                 bar_y + y_range*0.03, bar_y],
                color="k", lw=1.2)
        ax.text((x1+x2)/2, bar_y + y_range*0.04,
                p_label(p_val), ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_xticks(list(positions.values()))
    ax.set_xticklabels(tick_labels, fontsize=12)      
    ax.set_title(win_label, fontweight="bold", fontsize=12)
    ax.set_ylabel("PLF (0–1)", fontsize=12)            
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)

   
    n_val = len(vals[group_labels[0]])
    ax.text(0.98, 0.02, f"n={n_val}",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=10, color="black")

plt.suptitle("PLF", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(
    "/Users/deimantemickute/Desktop/Bakalauras/PLF_statistika.png",
    dpi=150, bbox_inches="tight")
plt.show()