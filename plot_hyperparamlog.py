import numpy as np
import matplotlib.pyplot as plt

def plot_lambda_curve(lambdas, accs, stds,
                      label="Performance", color="#0072B2", style="o-",
                      title="Effect of λ on Performance", logx=True):
    """
    Plot performance vs λ with error bars and automatically detect baseline at λ=0.

    Args:
        lambdas (array-like): λ values
        accs (array-like): performance means
        stds (array-like): performance stds
        label (str): legend label
        color (str): color for curve
        style (str): matplotlib line/marker style
        title (str): figure title
        logx (bool): whether to use log scale for λ
    """
    lambdas = np.array(lambdas)
    accs = np.array(accs)
    stds = np.array(stds)


    # === Main performance curve ===
    plt.errorbar(
        lambdas, accs, yerr=stds,
        fmt=style, capsize=5, lw=2.2, markersize=7,
        color=color, ecolor="lightgray", label=label
    )

    # === Automatically detect baseline at λ=0 ===
    if 0 in lambdas:
        idx = np.where(lambdas == 0)[0][0]
        baseline = accs[idx]
        baseline_std = stds[idx]

        plt.axhline(y=baseline, color=color, linestyle='--', linewidth=1.8)
        plt.fill_between(
            [min(lambdas), max(lambdas)],
            baseline - baseline_std, baseline + baseline_std,
            color=color, alpha=0.08
        )

        print(f"Detected baseline: λ=0 → {baseline:.2f} ± {baseline_std:.2f}")
    else:
        baseline = baseline_std = None
        print("No λ=0 baseline found — skipping baseline line.")

    # === Axis scaling and labels ===
    if logx:
        plt.xscale('log')

    plt.xlabel('λ (log scale)' if logx else 'λ', fontsize=12)
    plt.ylabel('Performance (%)', fontsize=12)
    plt.title(title, fontsize=14, weight='semibold')

    # === Style tweaks ===
    plt.grid(True, which='both', ls=':', lw=0.6, color='gray', alpha=0.2)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')
    ax.spines['left'].set_alpha(0.4)
    ax.spines['bottom'].set_alpha(0.4)
    plt.legend(frameon=False, fontsize=11)
    plt.tight_layout()

    return baseline, baseline_std


# ==== Example usage ====
if __name__ == "__main__":

    lambdas = np.array([0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100])
    perf = np.array([71.12, 71.94, 72.30, 72.48, 71.07, 72.06, 68.13, 66.99, 66.89, 66.23, 66.56, 67.08])
    std = np.array([2.55, 2.60, 2.70, 2.56, 2.06, 2.43, 3.74, 3.43, 2.96, 3.16, 2.75, 2.45])

    plt.figure(figsize=(7, 5))

    plot_lambda_curve(
        lambdas, perf, std,
        label="Without contrastive loss",
        color="blue",
        style="s--",
        title="Effect of λ on Performance (No Contrastive Loss)"
    )

    plt.show()
