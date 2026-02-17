import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # safe in headless mode


def make_beta_schedule_mlp(num_steps: int, schedule: str = "cosine") -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(1e-4, 2e-2, num_steps)
    if schedule == "cosine":
        steps = num_steps + 1
        t = torch.linspace(0, num_steps, steps) / num_steps
        s = 0.008
        alphas_cumprod = torch.cos(((t + s) / (1 + s)) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(1e-8, 0.999)
    raise ValueError(f"Unknown schedule: {schedule}")


def make_beta_schedule_scaled_fixed(
    num_steps: int,
    schedule: str = "cosine",
    alpha_bar_target: float = 1e-3,
) -> torch.Tensor:
    """
    Take your original beta schedule (cosine/linear) and scale it so that
    the final alpha_bar ≈ alpha_bar_target.

    This preserves the *shape* of the schedule but makes the overall noise
    less or more aggressive depending on target.
    """
    betas_raw = make_beta_schedule_mlp(num_steps, schedule=schedule)
    betas_raw = betas_raw.clamp(1e-8, 0.999)

    def compute_alpha_bar(k: float) -> float:
        alphas = 1 - k * betas_raw
        alphas = alphas.clamp(1e-8, 0.999)
        ac = torch.cumprod(alphas, dim=0)[-1]
        return float(ac.item())

    # binary search on scale k
    low, high = 0.0, 10.0
    for _ in range(60):
        mid = 0.5 * (low + high)
        if compute_alpha_bar(mid) > alpha_bar_target:
            # too little noise -> increase k
            low = mid
        else:
            # too much noise -> decrease k
            high = mid
    k = 0.5 * (low + high)

    betas = (k * betas_raw).clamp(1e-8, 0.999)
    return betas


def inspect_schedule_scaled(num_steps: int = 1000,
                            alpha_bar_target: float = 1e-3,
                            save_path: str = "schedule_scaled_summary.png"):
    # --- build scaled schedules ---
    betas_cos = make_beta_schedule_scaled_fixed(num_steps, "cosine", alpha_bar_target)
    betas_lin = make_beta_schedule_scaled_fixed(num_steps, "linear", alpha_bar_target)

    alphas_cos = 1.0 - betas_cos
    alphas_lin = 1.0 - betas_lin

    ac_cos = torch.cumprod(alphas_cos, dim=0)
    ac_lin = torch.cumprod(alphas_lin, dim=0)

    noise_cos = torch.sqrt(1.0 - ac_cos)
    noise_lin = torch.sqrt(1.0 - ac_lin)

    snr_cos = ac_cos / (1.0 - ac_cos + 1e-8)
    snr_lin = ac_lin / (1.0 - ac_lin + 1e-8)

    timesteps = torch.arange(num_steps)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # 1) alpha_bar (cumulative)
    axs[0, 0].set_yscale("log")
    axs[0, 0].plot(timesteps, ac_cos, label="cosine ᾱₜ")
    axs[0, 0].plot(timesteps, ac_lin, label="linear ᾱₜ")
    axs[0, 0].set_title(f"ᾱₜ (log), target final ≈ {alpha_bar_target}")
    axs[0, 0].set_xlabel("timestep")
    axs[0, 0].set_ylabel("ᾱₜ")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # 2) noise std
    axs[0, 1].plot(timesteps, noise_cos, label="cosine noise std")
    axs[0, 1].plot(timesteps, noise_lin, label="linear noise std")
    axs[0, 1].set_title("Noise std = √(1 − ᾱₜ)")
    axs[0, 1].set_xlabel("timestep")
    axs[0, 1].set_ylabel("std")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # 3) SNR
    axs[1, 0].set_yscale("log")
    axs[1, 0].plot(timesteps, snr_cos, label="cosine SNR")
    axs[1, 0].plot(timesteps, snr_lin, label="linear SNR")
    axs[1, 0].set_title("SNR = ᾱₜ / (1 − ᾱₜ) (log)")
    axs[1, 0].set_xlabel("timestep")
    axs[1, 0].set_ylabel("SNR")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # 4) betas
    axs[1, 1].plot(timesteps, betas_cos, label="cosine βₜ")
    axs[1, 1].plot(timesteps, betas_lin, label="linear βₜ")
    axs[1, 1].set_title("βₜ")
    axs[1, 1].set_xlabel("timestep")
    axs[1, 1].set_ylabel("βₜ")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    # numeric summary
    sample_ts = [0, num_steps // 4, num_steps // 2, 3 * num_steps // 4, num_steps - 1]
    print(f"Saved figure to: {save_path}")
    print("Sample ᾱₜ values (cosine, scaled):")
    for t in sample_ts:
        print(f"  t={t:4d} -> ᾱₜ ≈ {ac_cos[t].item():.3e}")
    print("Final alpha_bar (cosine):", ac_cos[-1].item())
    print("\nSample ᾱₜ values (linear, scaled):")
    for t in sample_ts:
        print(f"  t={t:4d} -> ᾱₜ ≈ {ac_lin[t].item():.3e}")
    print("Final alpha_bar (linear):", ac_lin[-1].item())
inspect_schedule_scaled()