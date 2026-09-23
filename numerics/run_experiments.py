"""Reproduce the tuned numerical comparisons in Section 5.

The script compares overdamped, underdamped, and third-order Langevin dynamics
under a common fixed-friction/vanishing-noise convention.  It produces

  * figures/parameter_tuning.png
  * figures/annealing_three_way_comparison.png
  * numerics/results.json

Run ``python numerics/run_experiments.py`` for the paper settings or add
``--quick`` for a small smoke test.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
RESULTS_PATH = ROOT / "numerics" / "results.json"

METHODS = ("overdamped", "underdamped", "third_order")
ANNEAL_CONFINEMENT = 0.1
ANNEAL_LEFT_DEPTH = 0.65
ANNEAL_RIGHT_DEPTH = 0.45
ANNEAL_WELL_OFFSET = 2.0
LABELS = {
    "overdamped": "Overdamped",
    "underdamped": "Underdamped (UBU)",
    "third_order": "Third-order midpoint",
}
COLORS = {
    "overdamped": "#4477AA",
    "underdamped": "#EE6677",
    "third_order": "#228833",
}


def annealing_potential(x: np.ndarray) -> np.ndarray:
    """Smooth confining tilted double well used for annealing."""
    y = x[..., 0]
    c = ANNEAL_WELL_OFFSET
    return (
        0.5 * ANNEAL_CONFINEMENT * y**2
        - ANNEAL_LEFT_DEPTH * np.exp(-(y + c) ** 2)
        - ANNEAL_RIGHT_DEPTH * np.exp(-(y - c) ** 2)
    )


def annealing_gradient(x: np.ndarray) -> np.ndarray:
    y = x[..., 0]
    c = ANNEAL_WELL_OFFSET
    grad = np.empty_like(x, dtype=float)
    grad[..., 0] = (
        ANNEAL_CONFINEMENT * y
        + 2.0 * ANNEAL_LEFT_DEPTH * (y + c) * np.exp(-(y + c) ** 2)
        + 2.0 * ANNEAL_RIGHT_DEPTH * (y - c) * np.exp(-(y - c) ** 2)
    )
    return grad


def annealing_hessian(x: float) -> float:
    c = ANNEAL_WELL_OFFSET
    return (
        ANNEAL_CONFINEMENT
        + 2.0
        * ANNEAL_LEFT_DEPTH
        * (1.0 - 2.0 * (x + c) ** 2)
        * math.exp(-(x + c) ** 2)
        + 2.0
        * ANNEAL_RIGHT_DEPTH
        * (1.0 - 2.0 * (x - c) ** 2)
        * math.exp(-(x - c) ** 2)
    )


def find_annealing_landscape() -> dict[str, float]:
    """Locate the two minima and intervening saddle and return their depths."""
    grid = np.linspace(-8.0, 8.0, 4001)
    values = annealing_gradient(grid[:, None])[:, 0]
    roots: list[float] = []
    for left, right, g_left, g_right in zip(
        grid[:-1], grid[1:], values[:-1], values[1:]
    ):
        if g_left * g_right < 0.0:
            root = brentq(
                lambda y: float(annealing_gradient(np.array([[y]]))[0, 0]),
                left,
                right,
                xtol=1e-14,
            )
            if not roots or abs(root - roots[-1]) > 1e-8:
                roots.append(root)
    minima = [root for root in roots if annealing_hessian(root) > 0.0]
    saddles = [root for root in roots if annealing_hessian(root) < 0.0]
    if len(minima) != 2 or len(saddles) != 1:
        raise RuntimeError(f"unexpected annealing critical points: {roots}")
    minima.sort()
    left, right = minima
    saddle = saddles[0]
    u_left = float(annealing_potential(np.array([[left]]))[0])
    u_right = float(annealing_potential(np.array([[right]]))[0])
    u_saddle = float(annealing_potential(np.array([[saddle]]))[0])
    if not u_left < u_right < u_saddle:
        raise RuntimeError("tilted-well energy ordering failed")
    return {
        "global_minimizer": left,
        "local_minimizer": right,
        "saddle": saddle,
        "global_minimum": u_left,
        "local_minimum": u_right,
        "saddle_value": u_saddle,
        "local_to_global_gap": u_right - u_left,
        "critical_depth": u_saddle - u_right,
    }


def canonical_temperature(t: float | np.ndarray, energy: float, t0: float) -> np.ndarray:
    return energy / np.log(np.asarray(t) + t0)


def make_grid(
    *,
    h0: float,
    exponent: float,
    t0: float,
    n_steps: int | None = None,
    horizon: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return left endpoints tau and steps h for h_k=c(tau_k+t0)^(-a)."""
    if (n_steps is None) == (horizon is None):
        raise ValueError("specify exactly one of n_steps or horizon")
    c_h = h0 * t0**exponent
    tau_values: list[float] = []
    h_values: list[float] = []
    tau = 0.0
    while n_steps is None or len(h_values) < n_steps:
        if horizon is not None and tau >= horizon:
            break
        h = c_h * (tau + t0) ** (-exponent)
        if horizon is not None and tau + h > horizon:
            h = horizon - tau
        tau_values.append(tau)
        h_values.append(h)
        tau += h
    return np.asarray(tau_values), np.asarray(h_values)


_GL_NODES, _GL_WEIGHTS = leggauss(16)
_GL_U = 0.5 * (_GL_NODES + 1.0)
_GL_W = 0.5 * _GL_WEIGHTS


def third_order_coefficients(
    h: float, lam: float = 1.0, gamma: float = 1.0
) -> tuple[float, float, float, float, float, float, float, np.ndarray]:
    """Mean coefficients and a Cholesky factor of Sigma_h.

    The factor L satisfies L L^T = Sigma_h, where the actual endpoint noise
    covariance is epsilon * Sigma_h.  An anisotropic scaling is used before
    Cholesky factorization to avoid loss of precision as h decreases.
    """
    e = math.exp(-gamma * h)
    one_minus_e = -math.expm1(-gamma * h)
    a_z = one_minus_e / gamma
    b_z = (gamma * h + math.expm1(-gamma * h)) / gamma**2
    d_z = (a_z - h * e) / gamma
    a_v = -(lam / gamma) * (h - a_z)
    b_v = -(lam / gamma) * (0.5 * h**2 - b_z)
    d_v = -(lam / gamma) * (b_z - d_z)

    r = h * _GL_U
    er = np.exp(-gamma * r)
    om = -np.expm1(-gamma * r)
    kx = (lam / gamma) * (r + np.expm1(-gamma * r) / gamma)
    kv = (lam / gamma) * om
    kz = er - (lam**2 / gamma**2) * (om - gamma * r * er)
    kernels = np.stack((kx, kv, kz), axis=1)
    scales = np.array((h**2.5, h**1.5, h**0.5))
    scaled = kernels / scales
    gram = 2.0 * gamma * h * np.einsum(
        "q,qi,qj->ij", _GL_W, scaled, scaled
    )
    gram = 0.5 * (gram + gram.T)
    eig_min = float(np.linalg.eigvalsh(gram)[0])
    if eig_min <= 0.0:
        gram += (abs(eig_min) + 1e-14) * np.eye(3)
    chol = scales[:, None] * np.linalg.cholesky(gram)
    return a_z, a_v, b_z, b_v, d_z, d_v, e, chol


def third_order_step(
    x: np.ndarray,
    v: np.ndarray,
    z: np.ndarray,
    h: float,
    eps: float,
    grad: Callable[[np.ndarray], np.ndarray],
    rng: np.random.Generator,
    lam: float = 1.0,
    gamma: float = 1.0,
    coeffs: tuple[float, float, float, float, float, float, float, np.ndarray]
    | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if coeffs is None:
        coeffs = third_order_coefficients(h, lam, gamma)
    a_z, a_v, b_z, b_v, d_z, d_v, e, chol = coeffs
    force_integral = h * grad(x + 0.5 * h * v)

    mean_x = x + (h + lam * b_v) * v + lam * b_z * z - 0.5 * h * force_integral
    mean_v = (1.0 + lam * a_v) * v + lam * a_z * z - force_integral
    mean_z = (
        (-lam * a_z - lam**2 * d_v) * v
        + (e - lam**2 * d_z) * z
        + (lam * b_z / h) * force_integral
    )

    normals = rng.standard_normal(x.shape + (3,))
    noise = math.sqrt(eps) * np.einsum("ab,...b->...a", chol, normals)
    return mean_x + noise[..., 0], mean_v + noise[..., 1], mean_z + noise[..., 2]


def underdamped_free_flow_coefficients(
    h: float, eta: float
) -> tuple[float, float, float, float, float]:
    """Coefficients for the exact free underdamped flow over time ``h``.

    The returned lower-triangular factors generate the correlated position and
    velocity noise at unit temperature.  A series is used in the position
    variance to avoid cancellation on the decreasing annealing grid.
    """
    a = eta * h
    decay = math.exp(-a)
    one_minus_decay = -math.expm1(-a)
    one_minus_decay2 = -math.expm1(-2.0 * a)

    if abs(a) < 1e-3:
        bracket = a**3 / 3.0 - a**4 / 4.0 + 7.0 * a**5 / 60.0 - a**6 / 24.0
    else:
        bracket = a - 2.0 * one_minus_decay + 0.5 * one_minus_decay2
    var_x = 2.0 * bracket / eta**2
    cov_xv = one_minus_decay**2 / eta
    var_v = one_minus_decay2

    chol_x = math.sqrt(max(var_x, 0.0))
    chol_vx = cov_xv / chol_x
    chol_v = math.sqrt(max(var_v - chol_vx**2, 0.0))
    return decay, one_minus_decay / eta, chol_x, chol_vx, chol_v


def underdamped_free_flow_step(
    x: np.ndarray,
    v: np.ndarray,
    eps: float,
    rng: np.random.Generator,
    coeffs: tuple[float, float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one exact free underdamped Gaussian subflow."""
    decay, position_drift, chol_x, chol_vx, chol_v = coeffs
    normals = rng.standard_normal(x.shape + (2,))
    scale = math.sqrt(eps)
    noise_x = scale * chol_x * normals[..., 0]
    noise_v = scale * (
        chol_vx * normals[..., 0] + chol_v * normals[..., 1]
    )
    return x + position_drift * v + noise_x, decay * v + noise_v


def underdamped_ubu_step(
    x: np.ndarray,
    v: np.ndarray,
    h: float,
    eps: float,
    grad: Callable[[np.ndarray], np.ndarray],
    rng: np.random.Generator,
    eta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One UBU step with one force evaluation at the stochastic midpoint."""
    half_coeffs = underdamped_free_flow_coefficients(0.5 * h, eta)
    x, v = underdamped_free_flow_step(x, v, eps, rng, half_coeffs)
    v = v - h * grad(x)
    return underdamped_free_flow_step(x, v, eps, rng, half_coeffs)


def checkpoint_indices(n_steps: int, count: int = 100) -> np.ndarray:
    positive = np.unique(
        np.rint(np.geomspace(1, n_steps, count)).astype(int)
    )
    return np.unique(np.concatenate(([0], positive)))


def simulate_annealing(
    method: str,
    tau: np.ndarray,
    steps: np.ndarray,
    *,
    n_paths: int,
    energy: float,
    t0: float,
    u_star: float,
    start: float,
    delta: float,
    seed: int,
    eta: float = 1.0,
    lam: float = 1.0,
    gamma: float = 1.0,
) -> dict:
    rng = np.random.default_rng(seed)
    x = np.full((n_paths, 1), start)
    v = np.zeros_like(x)
    z = np.zeros_like(x)
    best = annealing_potential(x)
    hit_time = np.full(n_paths, np.inf)
    basin = np.ones(n_paths, dtype=np.int8)
    crossings = np.zeros(n_paths, dtype=np.int32)

    cp = checkpoint_indices(len(steps))
    cp_set = set(cp.tolist())
    records: dict[str, list[float]] = {
        "time": [],
        "terminal_failure": [],
        "best_failure": [],
        "mean_gap": [],
        "median_gap": [],
        "q25_gap": [],
        "q75_gap": [],
    }

    def record(index: int, time: float) -> None:
        values = annealing_potential(x)
        gaps = values - u_star
        records["time"].append(float(time))
        records["terminal_failure"].append(float(np.mean(gaps > delta)))
        records["best_failure"].append(float(np.mean(best - u_star > delta)))
        records["mean_gap"].append(float(np.mean(gaps)))
        records["median_gap"].append(float(np.median(gaps)))
        records["q25_gap"].append(float(np.quantile(gaps, 0.25)))
        records["q75_gap"].append(float(np.quantile(gaps, 0.75)))

    record(0, 0.0)
    for k, (time, h) in enumerate(zip(tau, steps), start=1):
        eps = float(canonical_temperature(time, energy, t0))
        if method == "overdamped":
            x += -h * annealing_gradient(x) + math.sqrt(2.0 * eps * h) * rng.standard_normal(x.shape)
        elif method == "underdamped":
            x, v = underdamped_ubu_step(
                x,
                v,
                float(h),
                eps,
                annealing_gradient,
                rng,
                eta,
            )
        elif method == "third_order":
            coeffs = third_order_coefficients(float(h), lam, gamma)
            x, v, z = third_order_step(
                x,
                v,
                z,
                float(h),
                eps,
                annealing_gradient,
                rng,
                lam,
                gamma,
                coeffs,
            )
        else:
            raise ValueError(method)

        values = annealing_potential(x)
        best = np.minimum(best, values)
        newly_hit = np.isinf(hit_time) & (values - u_star <= delta)
        hit_time[newly_hit] = time + h

        reached_left = (basin == 1) & (x[:, 0] < -1.0)
        reached_right = (basin == -1) & (x[:, 0] > 1.0)
        crossings[reached_left | reached_right] += 1
        basin[reached_left] = -1
        basin[reached_right] = 1

        if k in cp_set:
            record(k, time + h)

    final_time = float(tau[-1] + steps[-1])
    final_values = annealing_potential(x)
    terminal_success = final_values - u_star <= delta
    ever_success = np.isfinite(hit_time)
    rmst = np.minimum(hit_time, final_time)
    summary = {
        "method": method,
        "n_paths": n_paths,
        "n_steps": int(len(steps)),
        "physical_horizon": final_time,
        "terminal_temperature": float(canonical_temperature(final_time, energy, t0)),
        "terminal_success": float(np.mean(terminal_success)),
        "terminal_success_se": float(np.std(terminal_success, ddof=1) / math.sqrt(n_paths)),
        "ever_success": float(np.mean(ever_success)),
        "ever_success_se": float(np.std(ever_success, ddof=1) / math.sqrt(n_paths)),
        "restricted_mean_hitting_time": float(np.mean(rmst)),
        "restricted_mean_hitting_time_se": float(np.std(rmst, ddof=1) / math.sqrt(n_paths)),
        "mean_terminal_gap": float(np.mean(final_values - u_star)),
        "mean_terminal_gap_se": float(np.std(final_values - u_star, ddof=1) / math.sqrt(n_paths)),
        "median_terminal_gap": float(np.median(final_values - u_star)),
        "mean_completed_basin_crossings": float(np.mean(crossings)),
        "mean_completed_basin_crossings_se": float(np.std(crossings, ddof=1) / math.sqrt(n_paths)),
    }
    return {"summary": summary, "curves": records}


def plot_parameter_tuning(annealing_tuning: dict) -> None:
    """Plot annealing validation scores used before held-out evaluation."""
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))

    ud_rows = sorted(
        (row for row in annealing_tuning["results"] if row["method"] == "underdamped"),
        key=lambda row: row["eta"],
    )
    axes[0].errorbar(
        [row["eta"] for row in ud_rows],
        [row["terminal_success"] for row in ud_rows],
        yerr=[row["terminal_success_se"] for row in ud_rows],
        marker="o",
        capsize=3,
        color=COLORS["underdamped"],
    )
    selected_eta = annealing_tuning["selected"]["underdamped"]["eta"]
    axes[0].axvline(selected_eta, color="black", linestyle=":", alpha=0.7)
    axes[0].set_xscale("log")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_xlabel(r"UD friction $\eta$")
    axes[0].set_ylabel("Validation terminal success")
    axes[0].set_title("Underdamped UBU annealing")
    axes[0].grid(alpha=0.22, which="both")

    third_rows = [
        row for row in annealing_tuning["results"] if row["method"] == "third_order"
    ]
    lambdas = annealing_tuning["lambda_grid"]
    gammas = annealing_tuning["gamma_grid"]
    score = np.full((len(lambdas), len(gammas)), np.nan)
    for row in third_rows:
        i = lambdas.index(row["lambda"])
        j = gammas.index(row["gamma"])
        score[i, j] = row["terminal_success"]
    image = axes[1].imshow(score, origin="lower", vmin=0.0, vmax=1.0, aspect="auto")
    axes[1].set_xticks(np.arange(len(gammas)), [f"{value:g}" for value in gammas])
    axes[1].set_yticks(np.arange(len(lambdas)), [f"{value:g}" for value in lambdas])
    axes[1].set_xlabel(r"TOLD thermostat $\gamma$")
    axes[1].set_ylabel(r"TOLD coupling $\lambda$")
    axes[1].set_title("Third-order annealing")
    selected_third = annealing_tuning["selected"]["third_order"]
    axes[1].scatter(
        gammas.index(selected_third["gamma"]),
        lambdas.index(selected_third["lambda"]),
        marker="x",
        s=90,
        linewidths=2.5,
        color="white",
    )
    fig.colorbar(image, ax=axes[1], label="Validation terminal success")

    fig.tight_layout()
    fig.savefig(FIGURES / "parameter_tuning.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_annealing(common: dict[str, dict], budget: dict[str, dict], delta: float) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.0))
    for method in METHODS:
        curves = common[method]["curves"]
        time = np.asarray(curves["time"])
        terminal = 1.0 - np.asarray(curves["terminal_failure"])
        best = 1.0 - np.asarray(curves["best_failure"])
        axes[0].plot(time[1:], terminal[1:], color=COLORS[method], label=LABELS[method])
        axes[1].plot(time[1:], best[1:], color=COLORS[method], label=LABELS[method])

    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set_ylim(bottom=0.0)
    axes[1].set_ylim(0.0, 1.0)
    axes[0].set_title("Terminal success")
    axes[1].set_title("Ever-reached success")
    axes[0].set_ylabel("Estimated success probability")
    for ax in axes[:2]:
        ax.set_xlabel("Physical time")
        ax.grid(alpha=0.22, which="both")
    axes[0].legend(frameon=False)

    xloc = np.arange(len(METHODS))
    width = 0.36
    terminal_success = [budget[m]["summary"]["terminal_success"] for m in METHODS]
    ever_success = [budget[m]["summary"]["ever_success"] for m in METHODS]
    terminal_se = [budget[m]["summary"]["terminal_success_se"] for m in METHODS]
    ever_se = [budget[m]["summary"]["ever_success_se"] for m in METHODS]
    axes[2].bar(xloc - width / 2, terminal_success, width, yerr=terminal_se, capsize=3, label="Terminal")
    axes[2].bar(xloc + width / 2, ever_success, width, yerr=ever_se, capsize=3, label="Ever reached")
    axes[2].set_xticks(xloc, ("OD", "UD", "Third"))
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_ylabel("Success probability")
    axes[2].set_title("Equal gradient budget")
    axes[2].legend(frameon=False)
    axes[2].grid(alpha=0.22, axis="y")
    fig.suptitle(rf"Theory-guided annealing, objective tolerance $\delta={delta:g}$", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "annealing_three_way_comparison.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small smoke-test run")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="regenerate figures from the existing numerics/results.json",
    )
    args = parser.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        payload = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        plot_annealing(
            payload["annealing_common_horizon"],
            payload["annealing_equal_budget"],
            payload["settings"]["objective_tolerance"],
        )
        plot_parameter_tuning(payload["annealing_parameter_tuning"])
        print("regenerated figures from stored results", flush=True)
        return

    if args.quick:
        n_anneal, n_anneal_tune = 300, 120
        common_horizon = 30.0
        budget_steps = 4000
        refinement_paths = 150
    else:
        n_anneal, n_anneal_tune = 4000, 1200
        common_horizon = 300.0
        budget_steps = 40000
        refinement_paths = 1000

    t0 = math.e
    landscape = find_annealing_landscape()
    energy = 0.28
    if energy <= landscape["critical_depth"]:
        raise RuntimeError("cooling energy must exceed the critical depth")
    delta = 0.08
    h0 = 0.05
    theta = landscape["critical_depth"] / energy
    third_budget_exponent = 0.25 * (1.0 + theta)
    annealing_eta_grid = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
    annealing_lambda_grid = (0.25, 0.5, 1.0, 1.5, 2.0)
    annealing_gamma_grid = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)

    u_star = landscape["global_minimum"]
    annealing_start = landscape["local_minimizer"]
    print("annealing landscape", landscape, flush=True)

    common_tau, common_steps = make_grid(
        h0=h0, exponent=0.5, t0=t0, horizon=common_horizon
    )

    # Tune method-specific friction/coupling on validation paths under the
    # theorem's canonical E/log(t+t0) schedule.  Terminal success is primary.
    annealing_tuning_results: list[dict] = []
    od_validation = simulate_annealing(
        "overdamped",
        common_tau,
        common_steps,
        n_paths=n_anneal_tune,
        energy=energy,
        t0=t0,
        u_star=u_star,
        start=annealing_start,
        delta=delta,
        seed=41000,
    )["summary"]
    annealing_tuning_results.append(od_validation)
    print("annealing tuning", od_validation, flush=True)
    for eta_candidate in annealing_eta_grid:
        summary = simulate_annealing(
            "underdamped",
            common_tau,
            common_steps,
            n_paths=n_anneal_tune,
            energy=energy,
            t0=t0,
            u_star=u_star,
            start=annealing_start,
            delta=delta,
            seed=41001,
            eta=eta_candidate,
        )["summary"]
        summary["eta"] = eta_candidate
        annealing_tuning_results.append(summary)
        print("annealing tuning", summary, flush=True)
    for lam_candidate in annealing_lambda_grid:
        for gamma_candidate in annealing_gamma_grid:
            summary = simulate_annealing(
                "third_order",
                common_tau,
                common_steps,
                n_paths=n_anneal_tune,
                energy=energy,
                t0=t0,
                u_star=u_star,
                start=annealing_start,
                delta=delta,
                seed=41002,
                lam=lam_candidate,
                gamma=gamma_candidate,
            )["summary"]
            summary["lambda"] = lam_candidate
            summary["gamma"] = gamma_candidate
            annealing_tuning_results.append(summary)
            print("annealing tuning", summary, flush=True)

    selected_ud = max(
        (row for row in annealing_tuning_results if row["method"] == "underdamped"),
        key=lambda row: (
            row["terminal_success"],
            row["ever_success"],
            -row["restricted_mean_hitting_time"],
        ),
    )
    selected_third = max(
        (row for row in annealing_tuning_results if row["method"] == "third_order"),
        key=lambda row: (
            row["terminal_success"],
            row["ever_success"],
            -row["restricted_mean_hitting_time"],
        ),
    )
    annealing_eta = selected_ud["eta"]
    annealing_lam = selected_third["lambda"]
    annealing_gamma = selected_third["gamma"]

    common_results: dict[str, dict] = {}
    for method_id, method in enumerate(METHODS):
        result = simulate_annealing(
            method,
            common_tau,
            common_steps,
            n_paths=n_anneal,
            energy=energy,
            t0=t0,
            u_star=u_star,
            start=annealing_start,
            delta=delta,
            seed=51000 + method_id,
            eta=annealing_eta,
            lam=annealing_lam,
            gamma=annealing_gamma,
        )
        common_results[method] = result
        print("common", result["summary"], flush=True)

    # Equal force-evaluation budget.  OD and UD use the standard a=1/2 grid;
    # third order uses a_*=(1+D/E)/4 from Theorem 4.1.
    budget_results: dict[str, dict] = {}
    for method_id, method in enumerate(METHODS):
        exponent = third_budget_exponent if method == "third_order" else 0.5
        tau, steps = make_grid(
            h0=h0, exponent=exponent, t0=t0, n_steps=budget_steps
        )
        result = simulate_annealing(
            method,
            tau,
            steps,
            n_paths=n_anneal,
            energy=energy,
            t0=t0,
            u_star=u_star,
            start=annealing_start,
            delta=delta,
            seed=61000 + method_id,
            eta=annealing_eta,
            lam=annealing_lam,
            gamma=annealing_gamma,
        )
        budget_results[method] = result
        print("budget", result["summary"], flush=True)

    # Check that the common-horizon ranking is stable after halving h0.
    refined_tau, refined_steps = make_grid(
        h0=0.5 * h0, exponent=0.5, t0=t0, horizon=common_horizon
    )
    refinement_results: dict[str, dict] = {}
    for method_id, method in enumerate(METHODS):
        result = simulate_annealing(
            method,
            refined_tau,
            refined_steps,
            n_paths=refinement_paths,
            energy=energy,
            t0=t0,
            u_star=u_star,
            start=annealing_start,
            delta=delta,
            seed=71000 + method_id,
            eta=annealing_eta,
            lam=annealing_lam,
            gamma=annealing_gamma,
        )
        refinement_results[method] = result
        print("refined", result["summary"], flush=True)

    plot_annealing(common_results, budget_results, delta)
    annealing_tuning = {
        "n_paths": n_anneal_tune,
        "eta_grid": list(annealing_eta_grid),
        "lambda_grid": list(annealing_lambda_grid),
        "gamma_grid": list(annealing_gamma_grid),
        "selection_objective": "maximum validation terminal success, with ever-reached success and restricted mean hitting time as tie-breakers",
        "selected": {
            "underdamped": {"eta": annealing_eta},
            "third_order": {"lambda": annealing_lam, "gamma": annealing_gamma},
        },
        "results": annealing_tuning_results,
    }
    plot_parameter_tuning(annealing_tuning)

    payload = {
        "settings": {
            "quick": args.quick,
            "annealing_parameters": {
                "eta": annealing_eta,
                "lambda": annealing_lam,
                "gamma": annealing_gamma,
            },
            "temperature_energy": energy,
            "temperature_shift": t0,
            "critical_depth": landscape["critical_depth"],
            "depth_ratio": theta,
            "third_order_budget_step_exponent": third_budget_exponent,
            "objective_tolerance": delta,
            "initial_step": h0,
            "integrators": {
                "overdamped": "Euler--Maruyama",
                "underdamped": "UBU midpoint splitting",
                "third_order": "Mou et al. Algorithm 1 with midpoint force integral",
            },
            "annealing_landscape": landscape,
        },
        "annealing_parameter_tuning": annealing_tuning,
        "annealing_common_horizon": common_results,
        "annealing_equal_budget": budget_results,
        "annealing_step_refinement": refinement_results,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
