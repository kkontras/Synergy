#!/usr/bin/env python3
"""
Waveform-to-PNG (pretty, no axes, big label)

Edit the two variables below, then run:
  python waveform_png.py
"""

# ====== PSEUDO-INSERTIONS (EDIT THESE) ======
WAV_PATH = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/CremaD/CREMA-D/AudioWAV/1091_IOM_DIS_XX.wav"
LABEL_TEXT = "Disgust"
WAV_PATH = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/CremaD/CREMA-D/AudioWAV/1091_IOM_HAP_XX.wav"
LABEL_TEXT = "Happy"
# ===========================================

# Optional: output path (auto if left as None)
OUT_PNG_PATH = None  # e.g. "/path/to/output.png" or None to auto

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def _read_wav(path: str):
    """
    Read wav without extra dependencies.
    Supports PCM 8/16/24/32-bit and float in common cases.
    """
    import wave
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        fr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 1:
        x = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        x = (x - 128.0) / 128.0
    elif sampwidth == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        x /= 32768.0
    elif sampwidth == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        x = (b[:, 0].astype(np.int32) |
             (b[:, 1].astype(np.int32) << 8) |
             (b[:, 2].astype(np.int32) << 16))
        x = (x ^ 0x800000) - 0x800000
        x = x.astype(np.float32) / 8388608.0
    elif sampwidth == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
        x /= 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth} bytes")

    if n_channels > 1:
        x = x.reshape(-1, n_channels).mean(axis=1)

    return x, fr

def _downsample_for_plot(y: np.ndarray, target_points: int = 4000) -> np.ndarray:
    n = y.size
    if n <= target_points:
        return y

    bins = target_points // 2
    edges = np.linspace(0, n, bins + 1, dtype=np.int64)
    mins = np.empty(bins, dtype=np.float32)
    maxs = np.empty(bins, dtype=np.float32)

    for i in range(bins):
        seg = y[edges[i]:edges[i + 1]]
        if seg.size == 0:
            mins[i] = 0.0
            maxs[i] = 0.0
        else:
            mins[i] = float(seg.min())
            maxs[i] = float(seg.max())

    out = np.empty(bins * 2, dtype=np.float32)
    out[0::2] = mins
    out[1::2] = maxs
    return out

def render_waveform_png(wav_path: str, label: str, out_png_path: str | None = None):
    y, sr = _read_wav(wav_path)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    # Gentle dynamic compression for nicer visuals
    peak = np.max(np.abs(y)) + 1e-12
    y = y / peak
    y = np.tanh(1.6 * y)
    y = y / (np.max(np.abs(y)) + 1e-12)

    y_plot = _downsample_for_plot(y, target_points=5000)
    x_plot = np.linspace(0, 1, y_plot.size, dtype=np.float32)

    # ---- Style (bright bg, lighter wave, more opacity layering) ----
    bg = "lightgray"          # bright sky-blue background
    bg2 = "lightgray"         # slightly deeper for gradient (still bright)
    wave = "#EAF7FF"        # lighter than before (soft icy white-blue)
    glow = "#2B78FF"        # deeper blue glow
    text = "darkgray"        # very light label
    shadow = "#001018"

    wave = "#071826"          # deep blue background
    # wave = "#7CC7FF"          # bright sky-blue waveform
    # glow = "#2B78FF"        # deeper blue for subtle glow
    # text = "#E6F3FF"        # near-white blue


    fig = plt.figure(figsize=(5, 3), dpi=220, facecolor=bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.05, 1.05)

    # Background gradient (subtle but slick)
    cmap = LinearSegmentedColormap.from_list("bggrad", [bg2, bg])
    grad = np.linspace(0, 1, 512).reshape(512, 1)
    ax.imshow(
        grad,
        extent=[0, 1, -1.05, 1.05],
        origin="lower",
        aspect="auto",
        cmap=cmap,
        alpha=0.4,
        zorder=0,
    )

    # Very soft vignette / haze (adds depth)
    ax.fill_between([0, 1], [-1.05, -1.05], [1.05, 1.05], color="#0B2540", alpha=0.06, zorder=0)

    # Wave: glow layers (more opacity-based “cool” look)
    ax.plot(x_plot, y_plot, color=glow, linewidth=7.0, alpha=0.14, solid_capstyle="round", zorder=2)
    ax.plot(x_plot, y_plot, color=glow, linewidth=4.0, alpha=0.18, solid_capstyle="round", zorder=3)
    ax.plot(x_plot, y_plot, color=wave, linewidth=2.3, alpha=0.92, solid_capstyle="round", zorder=4)

    # Fill under waveform (more opacity, but still classy)
    ax.fill_between(x_plot, 0, y_plot, color=wave, alpha=0.16, linewidth=0, zorder=1)

    # Big label (increase font-size)
    # shadow first (slight offset)
    # ax.text(
    #     0.35, 0.218, label,
    #     transform=ax.transAxes,
    #     ha="left", va="top",
    #     fontsize=34, fontweight="bold",
    #     color=shadow, alpha=0.45,
    #     zorder=9,
    # )
    # main text
    # ax.text(
    #     0.03, 0.8, label,
    #     transform=ax.transAxes,
    #     ha="left", va="center",
    #     fontsize=54, fontweight="bold",
    #     color=text, alpha=0.98,
    #     zorder=10,
    #     bbox=dict(
    #         boxstyle="round,pad=0.35,rounding_size=0.2",
    #         facecolor="gray",  # a bit darker than #7CC7FF
    #         alpha=0.60
    #     ),
    #     path_effects=[
    #         __import__("matplotlib.patheffects").patheffects.withStroke(
    #             linewidth=3.5, foreground="black", alpha=0.55
    #         )
    #     ],
    # )

    import matplotlib.patheffects as pe

    ax.text(
        0.03, 0.85, label,
        transform=ax.transAxes,
        ha="left", va="center",
        fontsize=32, fontweight="bold",
        color="#0B2540",  # darker fill for the letters
        alpha=0.98,
        zorder=10,
        bbox=dict(
            boxstyle="round,pad=0.35,rounding_size=0.2",
            facecolor="gray",  # mid blue, slightly darker than bg
            edgecolor="#EAF7FF",  # light border around the box
            linewidth=2.0,
            alpha=0.42
        ),
        path_effects=[
            pe.withStroke(
                linewidth=3.5,
                foreground="#EAF7FF",  # light outline on letters
                alpha=0.85
            )
        ],
    )


    if out_png_path is None:
        base = os.path.splitext(os.path.basename(wav_path))[0]
        safe_label = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in label.strip())
        out_png_path = f"{base}__{safe_label}.png" if safe_label else f"{base}.png"

    fig.savefig(out_png_path, facecolor=bg, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    print(f"Saved: {out_png_path}")

if __name__ == "__main__":
    render_waveform_png(WAV_PATH, LABEL_TEXT, OUT_PNG_PATH)
