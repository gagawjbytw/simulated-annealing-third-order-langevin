"""High-dimensional nonconvex neural-network annealing experiment.

The potential is the regularized empirical risk of a two-layer tanh network,

    f_theta(x) = A / sqrt(m) * sum_j tanh(b_j) tanh(w_j^T x_tilde),

where both ``W`` and ``b`` are trainable.  The bounded parametrization of the
output weights and the quadratic regularizer make the example globally smooth
and dissipative while retaining the usual nonconvex hidden-unit geometry.

Run ``python numerics/run_neural_network_experiment.py`` for the paper settings
or add ``--quick`` for a smoke test.  The script writes

  * numerics/neural_network_results.json
  * arxiv/figures/neural_network_annealing.png

The finite neural-network problem has no certified global optimum.  Therefore,
success below means that a deterministic local optimization (a "quench") of an
annealing endpoint reaches the best basin found in a preliminary multistart
search.  Validation and evaluation paths use disjoint random seeds.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np
from scipy.optimize import minimize

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_experiments import (
    COLORS,
    LABELS,
    METHODS,
    canonical_temperature,
    checkpoint_indices,
    make_grid,
    third_order_coefficients,
    third_order_step,
    underdamped_ubu_step,
)


ROOT = Path(__file__).resolve().parents[1]
FIGURE_PATH = ROOT / "arxiv" / "figures" / "neural_network_annealing.png"
RESULTS_PATH = ROOT / "numerics" / "neural_network_results.json"


@dataclass(frozen=True)
class NetworkConfiguration:
    input_dimension: int = 20
    hidden_units: int = 4
    teacher_hidden_units: int = 12
    sample_size: int = 32
    output_scale: float = 2.0
    regularization: float = 1.0e-3
    data_seed: int = 11

    @property
    def augmented_dimension(self) -> int:
        return self.input_dimension + 1

    @property
    def parameter_dimension(self) -> int:
        return self.hidden_units * (self.augmented_dimension + 1)


class TwoLayerTanhPotential:
    """Regularized two-layer tanh regression loss and its exact gradient."""

    def __init__(self, config: NetworkConfiguration):
        self.config = config
        rng = np.random.default_rng(config.data_seed)
        features = rng.standard_normal(
            (config.sample_size, config.input_dimension)
        ) / math.sqrt(config.input_dimension)
        self.features = np.column_stack((features, np.ones(config.sample_size)))
        teacher_w = 1.5 * rng.standard_normal(
            (config.teacher_hidden_units, config.augmented_dimension)
        )
        teacher_b = 1.3 * rng.standard_normal(config.teacher_hidden_units)
        teacher_hidden = np.tanh(self.features @ teacher_w.T)
        self.targets = (
            config.output_scale
            / math.sqrt(config.teacher_hidden_units)
            * (teacher_hidden @ np.tanh(teacher_b))
        )

    def unpack(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        shape = theta.shape[:-1]
        split = self.config.hidden_units * self.config.augmented_dimension
        weights = theta[..., :split].reshape(
            shape + (self.config.hidden_units, self.config.augmented_dimension)
        )
        output_parameters = theta[..., split:]
        return weights, output_parameters

    def values(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        weights, output_parameters = self.unpack(theta)
        preactivation = np.einsum("nd,...md->...nm", self.features, weights)
        hidden = np.tanh(preactivation)
        output_weights = np.tanh(output_parameters)
        predictions = (
            self.config.output_scale
            / math.sqrt(self.config.hidden_units)
            * np.einsum("...nm,...m->...n", hidden, output_weights)
        )
        residual = predictions - self.targets
        return (
            0.5 * np.mean(residual**2, axis=-1)
            + 0.5 * self.config.regularization * np.sum(theta**2, axis=-1)
        )

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        weights, output_parameters = self.unpack(theta)
        preactivation = np.einsum("nd,...md->...nm", self.features, weights)
        hidden = np.tanh(preactivation)
        hidden_derivative = 1.0 - hidden**2
        output_weights = np.tanh(output_parameters)
        predictions = (
            self.config.output_scale
            / math.sqrt(self.config.hidden_units)
            * np.einsum("...nm,...m->...n", hidden, output_weights)
        )
        residual = predictions - self.targets
        scale = (
            self.config.output_scale
            / (math.sqrt(self.config.hidden_units) * self.config.sample_size)
        )
        grad_w = scale * np.einsum(
            "...n,...nm,...m,nd->...md",
            residual,
            hidden_derivative,
            output_weights,
            self.features,
        )
        grad_b = scale * np.einsum("...n,...nm->...m", residual, hidden)
        grad_b *= 1.0 - output_weights**2
        regularization = self.config.regularization
        return np.concatenate(
            (
                (grad_w + regularization * weights).reshape(
                    theta.shape[:-1] + (-1,)
                ),
                grad_b + regularization * output_parameters,
            ),
            axis=-1,
        )

    def value_and_gradient(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        return float(self.values(theta)), self.gradient(theta)


def local_quench(
    potential: TwoLayerTanhPotential,
    theta: np.ndarray,
    *,
    maxiter: int = 800,
) -> tuple[np.ndarray, float, float, bool]:
    """Apply deterministic L-BFGS-B, returning point, value and gradient norm."""
    bounds = [(-20.0, 20.0)] * potential.config.parameter_dimension

    def optimize(initial: np.ndarray, iterations: int):
        return minimize(
            potential.value_and_gradient,
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "maxiter": iterations,
                "maxls": 50,
                "ftol": 1.0e-13,
                "gtol": 1.0e-8,
            },
        )

    result = optimize(np.asarray(theta, dtype=float), maxiter)
    point = np.asarray(result.x)
    gradient_norm = float(np.linalg.norm(potential.gradient(point)))
    if not result.success and gradient_norm > 2.0e-6:
        result = optimize(point, max(2000, 3 * maxiter))
        point = np.asarray(result.x)
        gradient_norm = float(np.linalg.norm(potential.gradient(point)))
    converged = bool(result.success or gradient_norm <= 2.0e-6)
    return point, float(potential.values(point)), gradient_norm, converged


def cluster_minima(
    minima: list[tuple[np.ndarray, float, float, bool]],
    *,
    tolerance: float = 1.0e-6,
) -> list[dict]:
    """Cluster converged multistart results by objective value."""
    ordered = sorted((item for item in minima if item[3]), key=lambda item: item[1])
    clusters: list[dict] = []
    for point, value, gradient_norm, _ in ordered:
        if not clusters or abs(value - clusters[-1]["mean_value"]) > tolerance:
            clusters.append(
                {
                    "values": [value],
                    "representative": point,
                    "representative_gradient_norm": gradient_norm,
                    "mean_value": value,
                }
            )
        else:
            cluster = clusters[-1]
            cluster["values"].append(value)
            cluster["mean_value"] = float(np.mean(cluster["values"]))
            if gradient_norm < cluster["representative_gradient_norm"]:
                cluster["representative"] = point
                cluster["representative_gradient_norm"] = gradient_norm
    return clusters


def find_reference_basins(
    potential: TwoLayerTanhPotential,
    *,
    n_starts: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    minima: list[tuple[np.ndarray, float, float, bool]] = []
    for _ in range(n_starts):
        initial = rng.standard_normal(potential.config.parameter_dimension)
        minima.append(local_quench(potential, initial))
    clusters = cluster_minima(minima)
    if len(clusters) < 2:
        raise RuntimeError(
            "multistart search did not identify two distinct local-minimum levels"
        )

    minimum_cluster_count = max(2, math.ceil(0.01 * n_starts))
    eligible = [
        cluster for cluster in clusters if len(cluster["values"]) >= minimum_cluster_count
    ]
    if len(eligible) < 2:
        eligible = clusters
    start_cluster = eligible[-1]
    best_cluster = clusters[0]
    return {
        "n_starts": n_starts,
        "n_converged": sum(int(item[3]) for item in minima),
        "cluster_tolerance": 1.0e-6,
        "clusters": [
            {
                "mean_value": float(cluster["mean_value"]),
                "minimum_value": float(min(cluster["values"])),
                "maximum_value": float(max(cluster["values"])),
                "count": len(cluster["values"]),
                "representative_gradient_norm": float(
                    cluster["representative_gradient_norm"]
                ),
            }
            for cluster in clusters
        ],
        "best_value": float(min(best_cluster["values"])),
        "best_point": best_cluster["representative"],
        "start_value": float(start_cluster["mean_value"]),
        "start_point": start_cluster["representative"],
        "start_cluster_count": len(start_cluster["values"]),
    }


def simulate(
    method: str,
    potential: TwoLayerTanhPotential,
    start: np.ndarray,
    tau: np.ndarray,
    steps: np.ndarray,
    *,
    n_paths: int,
    energy: float,
    temperature_shift: float,
    seed: int,
    eta: float,
    lam: float,
    gamma: float,
) -> dict:
    rng = np.random.default_rng(seed)
    theta = np.repeat(start[None, :], n_paths, axis=0)
    velocity = np.zeros_like(theta)
    auxiliary = np.zeros_like(theta)
    best_values = potential.values(theta)

    checkpoints = checkpoint_indices(len(steps), count=80)
    checkpoint_set = set(checkpoints.tolist())
    curves = {
        "time": [0.0],
        "mean_objective": [float(np.mean(best_values))],
        "median_objective": [float(np.median(best_values))],
        "mean_best_objective": [float(np.mean(best_values))],
    }

    for index, (time, step) in enumerate(zip(tau, steps), start=1):
        h = float(step)
        temperature = float(
            canonical_temperature(float(time), energy, temperature_shift)
        )
        if method == "overdamped":
            theta += (
                -h * potential.gradient(theta)
                + math.sqrt(2.0 * temperature * h)
                * rng.standard_normal(theta.shape)
            )
        elif method == "underdamped":
            theta, velocity = underdamped_ubu_step(
                theta,
                velocity,
                h,
                temperature,
                potential.gradient,
                rng,
                eta,
            )
        elif method == "third_order":
            coefficients = third_order_coefficients(h, lam, gamma)
            theta, velocity, auxiliary = third_order_step(
                theta,
                velocity,
                auxiliary,
                h,
                temperature,
                potential.gradient,
                rng,
                lam,
                gamma,
                coefficients,
            )
        else:
            raise ValueError(method)

        values = potential.values(theta)
        best_values = np.minimum(best_values, values)
        if index in checkpoint_set:
            curves["time"].append(float(time + h))
            curves["mean_objective"].append(float(np.mean(values)))
            curves["median_objective"].append(float(np.median(values)))
            curves["mean_best_objective"].append(float(np.mean(best_values)))

    final_values = potential.values(theta)
    return {
        "endpoints": theta,
        "curves": curves,
        "raw_summary": {
            "method": method,
            "n_paths": n_paths,
            "n_steps": int(len(steps)),
            "physical_horizon": float(tau[-1] + steps[-1]),
            "terminal_temperature": float(
                canonical_temperature(
                    float(tau[-1] + steps[-1]), energy, temperature_shift
                )
            ),
            "mean_terminal_objective": float(np.mean(final_values)),
            "mean_terminal_objective_se": float(
                np.std(final_values, ddof=1) / math.sqrt(n_paths)
            ),
            "median_terminal_objective": float(np.median(final_values)),
        },
    }


def quench_endpoints(
    potential: TwoLayerTanhPotential,
    endpoints: np.ndarray,
    *,
    best_reference: float,
    success_tolerance: float,
) -> dict:
    values = np.empty(len(endpoints))
    gradient_norms = np.empty(len(endpoints))
    converged = np.zeros(len(endpoints), dtype=bool)
    for index, endpoint in enumerate(endpoints):
        _, value, gradient_norm, did_converge = local_quench(
            potential, endpoint, maxiter=600
        )
        values[index] = value
        gradient_norms[index] = gradient_norm
        converged[index] = did_converge
    success = values <= best_reference + success_tolerance
    n_paths = len(endpoints)
    return {
        "quenched_objectives": values.tolist(),
        "n_quench_converged": int(np.sum(converged)),
        "maximum_quench_gradient_norm": float(np.max(gradient_norms)),
        "best_basin_success": float(np.mean(success)),
        "best_basin_success_se": float(
            np.std(success, ddof=1) / math.sqrt(n_paths)
        ),
        "mean_quenched_objective": float(np.mean(values)),
        "mean_quenched_objective_se": float(
            np.std(values, ddof=1) / math.sqrt(n_paths)
        ),
        "median_quenched_objective": float(np.median(values)),
        "minimum_quenched_objective": float(np.min(values)),
        "maximum_quenched_objective": float(np.max(values)),
    }


def run_protocol(
    potential: TwoLayerTanhPotential,
    reference: dict,
    *,
    tau: np.ndarray,
    steps: np.ndarray,
    n_paths: int,
    energy: float,
    temperature_shift: float,
    seed_base: int,
    eta: float,
    lam: float,
    gamma: float,
    success_tolerance: float,
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for method_id, method in enumerate(METHODS):
        simulation = simulate(
            method,
            potential,
            reference["start_point"],
            tau,
            steps,
            n_paths=n_paths,
            energy=energy,
            temperature_shift=temperature_shift,
            seed=seed_base + method_id,
            eta=eta,
            lam=lam,
            gamma=gamma,
        )
        print(
            f"{method}: simulated {n_paths} paths; quenching endpoints",
            flush=True,
        )
        quenched = quench_endpoints(
            potential,
            simulation.pop("endpoints"),
            best_reference=reference["best_value"],
            success_tolerance=success_tolerance,
        )
        simulation["summary"] = {**simulation.pop("raw_summary"), **quenched}
        results[method] = simulation
        summary = simulation["summary"]
        print(
            method,
            {
                "best_basin_success": summary["best_basin_success"],
                "best_basin_success_se": summary["best_basin_success_se"],
                "mean_quenched_objective": summary["mean_quenched_objective"],
                "mean_terminal_objective": summary["mean_terminal_objective"],
                "n_quench_converged": summary["n_quench_converged"],
            },
            flush=True,
        )
    return results


def select_common_temperature(
    potential: TwoLayerTanhPotential,
    reference: dict,
    *,
    energy_grid: tuple[float, ...],
    tau: np.ndarray,
    steps: np.ndarray,
    n_paths: int,
    temperature_shift: float,
    eta: float,
    lam: float,
    gamma: float,
    success_tolerance: float,
) -> dict:
    """Select a shared energy by pooled, method-symmetric validation success."""
    rows: list[dict] = []
    for energy_id, energy in enumerate(energy_grid):
        method_success: dict[str, float] = {}
        for method_id, method in enumerate(METHODS):
            simulation = simulate(
                method,
                potential,
                reference["start_point"],
                tau,
                steps,
                n_paths=n_paths,
                energy=energy,
                temperature_shift=temperature_shift,
                seed=31000 + 10 * energy_id + method_id,
                eta=eta,
                lam=lam,
                gamma=gamma,
            )
            quenched = quench_endpoints(
                potential,
                simulation["endpoints"],
                best_reference=reference["best_value"],
                success_tolerance=success_tolerance,
            )
            method_success[method] = quenched["best_basin_success"]
        row = {
            "energy": energy,
            "method_success": method_success,
            "pooled_success": float(np.mean(list(method_success.values()))),
        }
        rows.append(row)
        print("temperature validation", row, flush=True)
    selected = min(rows, key=lambda row: abs(row["pooled_success"] - 0.4))
    return {
        "energy_grid": list(energy_grid),
        "n_paths_per_method": n_paths,
        "selection_rule": "pooled success closest to 0.4",
        "selected_energy": selected["energy"],
        "results": rows,
    }


def score_parameter_candidate(
    method: str,
    potential: TwoLayerTanhPotential,
    reference: dict,
    *,
    tau: np.ndarray,
    steps: np.ndarray,
    n_paths: int,
    energy: float,
    temperature_shift: float,
    seed: int,
    eta: float,
    lam: float,
    gamma: float,
    success_tolerance: float,
) -> dict:
    """Evaluate one kinetic parameter candidate on validation paths."""
    simulation = simulate(
        method,
        potential,
        reference["start_point"],
        tau,
        steps,
        n_paths=n_paths,
        energy=energy,
        temperature_shift=temperature_shift,
        seed=seed,
        eta=eta,
        lam=lam,
        gamma=gamma,
    )
    quenched = quench_endpoints(
        potential,
        simulation["endpoints"],
        best_reference=reference["best_value"],
        success_tolerance=success_tolerance,
    )
    return {
        "best_basin_success": quenched["best_basin_success"],
        "best_basin_success_se": quenched["best_basin_success_se"],
        "mean_quenched_objective": quenched["mean_quenched_objective"],
        "mean_quenched_objective_se": quenched["mean_quenched_objective_se"],
        "n_quench_converged": quenched["n_quench_converged"],
    }


def tune_kinetic_parameters(
    potential: TwoLayerTanhPotential,
    reference: dict,
    *,
    tau: np.ndarray,
    steps: np.ndarray,
    energy: float,
    temperature_shift: float,
    screen_paths: int,
    confirmation_paths: int,
    success_tolerance: float,
) -> dict:
    """Two-stage held-out tuning with common random numbers within each stage."""
    eta_grid = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
    lambda_grid = (0.25, 0.5, 1.0, 1.5, 2.0)
    gamma_grid = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)

    def selection_key(row: dict) -> tuple[float, float]:
        return row["best_basin_success"], -row["mean_quenched_objective"]

    screening: dict[str, list[dict]] = {
        "underdamped": [],
        "third_order": [],
    }
    for eta in eta_grid:
        row = score_parameter_candidate(
            "underdamped",
            potential,
            reference,
            tau=tau,
            steps=steps,
            n_paths=screen_paths,
            energy=energy,
            temperature_shift=temperature_shift,
            seed=81001,
            eta=eta,
            lam=1.0,
            gamma=4.0,
            success_tolerance=success_tolerance,
        )
        row["eta"] = eta
        screening["underdamped"].append(row)
        print("parameter screening UBU", row, flush=True)
    for lam in lambda_grid:
        for gamma in gamma_grid:
            row = score_parameter_candidate(
                "third_order",
                potential,
                reference,
                tau=tau,
                steps=steps,
                n_paths=screen_paths,
                energy=energy,
                temperature_shift=temperature_shift,
                seed=81002,
                eta=0.25,
                lam=lam,
                gamma=gamma,
                success_tolerance=success_tolerance,
            )
            row["lambda"] = lam
            row["gamma"] = gamma
            screening["third_order"].append(row)
            print("parameter screening third order", row, flush=True)

    finalists = {
        method: sorted(rows, key=selection_key, reverse=True)[:3]
        for method, rows in screening.items()
    }
    confirmation: dict[str, list[dict]] = {
        "underdamped": [],
        "third_order": [],
    }
    for candidate in finalists["underdamped"]:
        eta = candidate["eta"]
        row = score_parameter_candidate(
            "underdamped",
            potential,
            reference,
            tau=tau,
            steps=steps,
            n_paths=confirmation_paths,
            energy=energy,
            temperature_shift=temperature_shift,
            seed=82001,
            eta=eta,
            lam=1.0,
            gamma=4.0,
            success_tolerance=success_tolerance,
        )
        row["eta"] = eta
        confirmation["underdamped"].append(row)
        print("parameter confirmation UBU", row, flush=True)
    for candidate in finalists["third_order"]:
        lam = candidate["lambda"]
        gamma = candidate["gamma"]
        row = score_parameter_candidate(
            "third_order",
            potential,
            reference,
            tau=tau,
            steps=steps,
            n_paths=confirmation_paths,
            energy=energy,
            temperature_shift=temperature_shift,
            seed=82002,
            eta=0.25,
            lam=lam,
            gamma=gamma,
            success_tolerance=success_tolerance,
        )
        row["lambda"] = lam
        row["gamma"] = gamma
        confirmation["third_order"].append(row)
        print("parameter confirmation third order", row, flush=True)

    selected_ud = max(confirmation["underdamped"], key=selection_key)
    selected_third = max(confirmation["third_order"], key=selection_key)
    return {
        "screening_paths_per_candidate": screen_paths,
        "confirmation_paths_per_candidate": confirmation_paths,
        "selection_rule": (
            "maximum confirmed best-basin success, with mean quenched "
            "objective as tie-breaker"
        ),
        "eta_grid": list(eta_grid),
        "lambda_grid": list(lambda_grid),
        "gamma_grid": list(gamma_grid),
        "screening": screening,
        "finalists": finalists,
        "confirmation": confirmation,
        "selected": {
            "underdamped": {"eta": selected_ud["eta"]},
            "third_order": {
                "lambda": selected_third["lambda"],
                "gamma": selected_third["gamma"],
            },
        },
    }


def plot_results(reference: dict, evaluation: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.7, 3.9))

    clusters = reference["clusters"]
    best = reference["best_value"]
    gaps = 1.0e3 * (np.asarray([row["mean_value"] for row in clusters]) - best)
    counts = np.asarray([row["count"] for row in clusters])
    distinct_gaps = np.diff(np.unique(gaps))
    bar_width = (
        0.55 * float(np.min(distinct_gaps[distinct_gaps > 0.0]))
        if np.any(distinct_gaps > 0.0)
        else 0.02
    )
    axes[0].bar(gaps, counts, width=bar_width, color="#999999")
    axes[0].set_xlabel(r"Local-minimum gap from best found ($\times 10^{-3}$)")
    axes[0].set_ylabel("Multistart count")
    axes[0].set_title("Nonconvex basin structure")
    axes[0].grid(alpha=0.2, axis="y")

    xloc = np.arange(len(METHODS))
    probabilities = [
        evaluation[method]["summary"]["best_basin_success"] for method in METHODS
    ]
    errors = [
        evaluation[method]["summary"]["best_basin_success_se"] for method in METHODS
    ]
    axes[1].bar(
        xloc,
        probabilities,
        yerr=errors,
        capsize=4,
        color=[COLORS[method] for method in METHODS],
    )
    axes[1].set_xticks(xloc, ("OD", "UD", "Third"))
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("Best-basin probability")
    axes[1].set_title("Held-out annealing paths")
    axes[1].grid(alpha=0.2, axis="y")

    for method in METHODS:
        values = np.sort(
            np.asarray(evaluation[method]["summary"]["quenched_objectives"])
        )
        objective_gaps = 1.0e3 * (values - best)
        cdf = np.arange(1, len(values) + 1) / len(values)
        axes[2].step(
            objective_gaps,
            cdf,
            where="post",
            color=COLORS[method],
            label=LABELS[method],
        )
    axes[2].set_xlabel(r"Quenched objective gap ($\times 10^{-3}$)")
    axes[2].set_ylabel("Empirical CDF")
    axes[2].set_title("Endpoint basin quality")
    axes[2].grid(alpha=0.2)
    axes[2].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=250, bbox_inches="tight")
    plt.close(fig)


def serializable_reference(reference: dict) -> dict:
    return {
        key: (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in reference.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small smoke-test run")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="regenerate the figure from numerics/neural_network_results.json",
    )
    args = parser.parse_args()

    if args.plot_only:
        payload = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        plot_results(payload["reference_basins"], payload["evaluation"])
        print(f"wrote {FIGURE_PATH}", flush=True)
        return

    config = NetworkConfiguration()
    potential = TwoLayerTanhPotential(config)
    if args.quick:
        n_reference_starts = 32
        n_validation_paths = 32
        n_parameter_screen_paths = 16
        n_parameter_confirmation_paths = 32
        n_evaluation_paths = 96
        n_refinement_paths = 48
    else:
        n_reference_starts = 96
        n_validation_paths = 96
        n_parameter_screen_paths = 64
        n_parameter_confirmation_paths = 192
        n_evaluation_paths = 768
        n_refinement_paths = 192

    temperature_shift = math.e
    h0 = 0.05
    exponent = 0.5
    horizon = 50.0
    eta = 0.25
    lam = 1.0
    gamma = 4.0
    success_tolerance = 1.0e-6

    print(
        f"network parameter dimension: {config.parameter_dimension}; "
        f"multistart search with {n_reference_starts} starts",
        flush=True,
    )
    reference = find_reference_basins(
        potential, n_starts=n_reference_starts, seed=21000
    )
    print(
        "reference basin levels",
        [(row["mean_value"], row["count"]) for row in reference["clusters"]],
        flush=True,
    )

    tau, steps = make_grid(
        h0=h0,
        exponent=exponent,
        t0=temperature_shift,
        horizon=horizon,
    )
    validation = select_common_temperature(
        potential,
        reference,
        energy_grid=(0.004, 0.008, 0.012, 0.016, 0.024),
        tau=tau,
        steps=steps,
        n_paths=n_validation_paths,
        temperature_shift=temperature_shift,
        eta=eta,
        lam=lam,
        gamma=gamma,
        success_tolerance=success_tolerance,
    )
    energy = validation["selected_energy"]
    print(f"selected common temperature energy E={energy}", flush=True)

    parameter_tuning = tune_kinetic_parameters(
        potential,
        reference,
        tau=tau,
        steps=steps,
        energy=energy,
        temperature_shift=temperature_shift,
        screen_paths=n_parameter_screen_paths,
        confirmation_paths=n_parameter_confirmation_paths,
        success_tolerance=success_tolerance,
    )
    eta = parameter_tuning["selected"]["underdamped"]["eta"]
    lam = parameter_tuning["selected"]["third_order"]["lambda"]
    gamma = parameter_tuning["selected"]["third_order"]["gamma"]
    print(
        f"selected kinetic parameters: eta={eta}, lambda={lam}, gamma={gamma}",
        flush=True,
    )

    evaluation = run_protocol(
        potential,
        reference,
        tau=tau,
        steps=steps,
        n_paths=n_evaluation_paths,
        energy=energy,
        temperature_shift=temperature_shift,
        seed_base=91000,
        eta=eta,
        lam=lam,
        gamma=gamma,
        success_tolerance=success_tolerance,
    )

    # Diagnostic only: evaluate the one-dimensional experiment's inherited
    # kinetic parameters with the same random-number streams as the tuned
    # held-out runs.  These results do not participate in parameter selection.
    inherited_parameter_comparison = {
        "underdamped": score_parameter_candidate(
            "underdamped",
            potential,
            reference,
            tau=tau,
            steps=steps,
            n_paths=n_evaluation_paths,
            energy=energy,
            temperature_shift=temperature_shift,
            seed=91001,
            eta=0.25,
            lam=1.0,
            gamma=4.0,
            success_tolerance=success_tolerance,
        ),
        "third_order": score_parameter_candidate(
            "third_order",
            potential,
            reference,
            tau=tau,
            steps=steps,
            n_paths=n_evaluation_paths,
            energy=energy,
            temperature_shift=temperature_shift,
            seed=91002,
            eta=0.25,
            lam=1.0,
            gamma=4.0,
            success_tolerance=success_tolerance,
        ),
    }
    inherited_parameter_comparison["parameters"] = {
        "eta": 0.25,
        "lambda": 1.0,
        "gamma": 4.0,
    }
    inherited_parameter_comparison["note"] = (
        "Diagnostic same-random-number comparison only; not used for selection"
    )
    print(
        "inherited-parameter comparison",
        inherited_parameter_comparison,
        flush=True,
    )

    refined_tau, refined_steps = make_grid(
        h0=0.5 * h0,
        exponent=exponent,
        t0=temperature_shift,
        horizon=horizon,
    )
    refinement = run_protocol(
        potential,
        reference,
        tau=refined_tau,
        steps=refined_steps,
        n_paths=n_refinement_paths,
        energy=energy,
        temperature_shift=temperature_shift,
        seed_base=101000,
        eta=eta,
        lam=lam,
        gamma=gamma,
        success_tolerance=success_tolerance,
    )

    plot_results(reference, evaluation)
    payload = {
        "settings": {
            "quick": args.quick,
            "network": {
                **config.__dict__,
                "parameter_dimension": config.parameter_dimension,
                "architecture": "two-layer tanh with bounded trainable output weights",
            },
            "schedule": "epsilon_t = E / log(t + e)",
            "temperature_energy": energy,
            "temperature_shift": temperature_shift,
            "horizon": horizon,
            "initial_step": h0,
            "step_exponent": exponent,
            "kinetic_parameters": {
                "eta": eta,
                "lambda": lam,
                "gamma": gamma,
            },
            "success_definition": (
                "L-BFGS-B quench reaches the best multistart basin within "
                f"{success_tolerance:g} objective units"
            ),
            "success_tolerance": success_tolerance,
        },
        "temperature_validation": validation,
        "kinetic_parameter_tuning": parameter_tuning,
        "inherited_parameter_comparison": inherited_parameter_comparison,
        "reference_basins": serializable_reference(reference),
        "evaluation": evaluation,
        "step_refinement": refinement,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {RESULTS_PATH}", flush=True)
    print(f"wrote {FIGURE_PATH}", flush=True)


if __name__ == "__main__":
    main()
