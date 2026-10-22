"""
Full diagnostics for an emcee run stored in chain.h5.

Deliberately does NOT import anything from your model/likelihood module
(mcmc_model.py) -- reading diagnostics from the backend file doesn't
require re-evaluating the likelihood, so this avoids pulling in the
heavy baccoemu/TensorFlow import just to make plots. If your parameter
names differ, just edit PARAM_NAMES below.
"""

import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt

PARAM_NAMES = ["Omegam", "w0", "s8", "mc1"]
BACKEND_FILE = "chain.h5"


# ---------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------
def load_backend(backend_file=BACKEND_FILE):
    backend = emcee.backends.HDFBackend(backend_file, read_only=True)
    print(f"Loaded backend: nwalkers={backend.shape[0]}, "
          f"ndim={backend.shape[1]}, iterations={backend.iteration}")
    return backend


# ---------------------------------------------------------------------
# Numeric diagnostics
# ---------------------------------------------------------------------
def report_acceptance_fraction(backend):
    """Fraction of proposed steps that were accepted, per walker, averaged.
    Healthy range is typically ~0.2-0.5 for emcee's default stretch move;
    much lower means walkers are rarely moving (posterior is hard to
    explore, or step size too large), much higher can mean under-exploration
    of the space (steps too small / correlated).
    """
    accepted = backend.accepted  # shape: (nwalkers,)
    iteration = backend.iteration
    frac = accepted / iteration
    print(f"\nAcceptance fraction: mean={frac.mean():.3f}, "
          f"min={frac.min():.3f}, max={frac.max():.3f}")
    if frac.mean() < 0.1:
        print("  -> Low acceptance fraction: walkers may be struggling "
              "to move efficiently through the posterior.")
    elif frac.mean() > 0.6:
        print("  -> High acceptance fraction: chain may be under-exploring "
              "(steps too correlated/small).")
    else:
        print("  -> Acceptance fraction is in the healthy range.")
    return frac


def report_autocorr_time(backend):
    """Same 50x-tau check as before, using the full stored chain."""
    chain = backend.get_chain()
    nsteps = chain.shape[0]
    try:
        tau = backend.get_autocorr_time(tol=0)  # tol=0: don't raise, just warn
        print(f"\nIntegrated autocorrelation time per parameter: {tau}")
        print(f"Chain length: {nsteps} steps, 50*tau_max = {50 * np.max(tau):.0f}")
        if nsteps >= 50 * np.max(tau):
            print("  -> Chain length satisfies the 50x tau rule of thumb.")
        else:
            print(f"  -> Still short by a factor of "
                  f"{50 * np.max(tau) / nsteps:.2f}x -- consider running longer.")
        return tau
    except emcee.autocorr.AutocorrError as e:
        print("\nAutocorrelation time estimate failed entirely:", e)
        return None


def report_gelman_rubin(backend):
    """Split-chain Gelman-Rubin R-hat, treating each walker's trajectory
    as an independent chain (standard approach for ensemble samplers).

    Interpretation: R-hat close to 1.0 (conventionally < 1.01, or a looser
    < 1.1 threshold is also commonly used) indicates the walkers agree
    with each other about the target distribution. Large R-hat (>1.1)
    means between-walker variance is large relative to within-walker
    variance -- a sign that different walkers are still exploring
    different regions, i.e. not yet converged.
    """
    chain = backend.get_chain()  # (nsteps, nwalkers, ndim)
    nsteps, nwalkers, ndim = chain.shape

    # Split each walker's chain in half to get 2*nwalkers "chains" --
    # standard trick to make R-hat sensitive to within-chain non-stationarity
    # too, not just between-walker disagreement.
    half = nsteps // 2
    first_half = chain[:half]
    second_half = chain[half:2 * half]
    split_chains = np.concatenate(
        [first_half.transpose(1, 0, 2), second_half.transpose(1, 0, 2)], axis=0
    )  # shape: (2*nwalkers, half, ndim)

    m, n, _ = split_chains.shape  # m = number of "chains", n = steps per chain

    rhat = np.empty(ndim)
    for i in range(ndim):
        chains_i = split_chains[:, :, i]  # (m, n)
        chain_means = chains_i.mean(axis=1)          # (m,)
        chain_vars = chains_i.var(axis=1, ddof=1)     # (m,)

        W = chain_vars.mean()                          # within-chain variance
        B = n * chain_means.var(ddof=1)                 # between-chain variance
        var_hat = ((n - 1) / n) * W + B / n
        rhat[i] = np.sqrt(var_hat / W) if W > 0 else np.nan

    print("\nGelman-Rubin R-hat per parameter:")
    for name, r in zip(PARAM_NAMES, rhat):
        flag = "OK" if r < 1.05 else ("borderline" if r < 1.1 else "NOT CONVERGED")
        print(f"  {name}: R-hat = {r:.4f}  [{flag}]")
    return rhat


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------
def plot_trace_full(backend, out_path="trace_plot.png"):
    chain = backend.get_chain()
    nsteps, nwalkers, ndim = chain.shape

    fig, axes = plt.subplots(ndim, 1, figsize=(10, 2.5 * ndim), sharex=True)
    if ndim == 1:
        axes = [axes]
    for i in range(ndim):
        axes[i].plot(chain[:, :, i], alpha=0.3, lw=0.5, color="C0")
        axes[i].set_ylabel(PARAM_NAMES[i])
    axes[-1].set_xlabel("step number")
    fig.suptitle("MCMC parameter traces (all walkers)", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved full trace plot to {out_path}")


def plot_trace_subset(backend, n_walkers_to_show=5, out_path="trace_subset.png", seed=0):
    """Same idea as plot_trace_full, but only a handful of walkers at
    higher opacity, so individual walker behavior is actually visible
    instead of being lost in overplotting.
    """
    chain = backend.get_chain()
    nsteps, nwalkers, ndim = chain.shape
    rng = np.random.default_rng(seed)
    idx = rng.choice(nwalkers, size=min(n_walkers_to_show, nwalkers), replace=False)

    fig, axes = plt.subplots(ndim, 1, figsize=(10, 2.5 * ndim), sharex=True)
    if ndim == 1:
        axes = [axes]
    for i in range(ndim):
        for w in idx:
            axes[i].plot(chain[:, w, i], alpha=0.8, lw=0.8)
        axes[i].set_ylabel(PARAM_NAMES[i])
    axes[-1].set_xlabel("step number")
    fig.suptitle(f"MCMC parameter traces ({len(idx)} walkers shown)", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved subset trace plot to {out_path}")


def plot_corner(flat_samples, out_path="corner_plot.png"):
    fig = corner.corner(
        flat_samples,
        labels=PARAM_NAMES,
        show_titles=True,
        title_fmt=".4f",
        quantiles=[0.16, 0.5, 0.84],
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved corner plot to {out_path}")


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------
def report_quantiles(flat_samples):
    print("\nParameter constraints (16th / 50th / 84th percentile):")
    for i, name in enumerate(PARAM_NAMES):
        q16, q50, q84 = np.percentile(flat_samples[:, i], [16, 50, 84])
        print(f"  {name} = {q50:.4f} (+{q84 - q50:.4f} / -{q50 - q16:.4f})")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def run_all_diagnostics(backend_file=BACKEND_FILE, burnin_frac=0.3, thin=None):
    backend = load_backend(backend_file)

    report_acceptance_fraction(backend)
    tau = report_autocorr_time(backend)
    report_gelman_rubin(backend)

    plot_trace_full(backend, out_path="trace_plot.png")
    plot_trace_subset(backend, n_walkers_to_show=5, out_path="trace_subset.png")

    nsteps = backend.get_chain().shape[0]
    burnin = int(burnin_frac * nsteps)
    if thin is None:
        thin = max(1, int(np.min(tau) / 2)) if tau is not None else 1

    flat_samples = backend.get_chain(discard=burnin, thin=thin, flat=True)
    print(f"\nFlattened chain shape after burn-in={burnin}, thin={thin}: "
          f"{flat_samples.shape}")

    report_quantiles(flat_samples)
    plot_corner(flat_samples, out_path="corner_plot.png")

    return backend, flat_samples


if __name__ == "__main__":
    backend, flat_samples = run_all_diagnostics(BACKEND_FILE)
