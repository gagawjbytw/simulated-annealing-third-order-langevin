"""Real-data annealing experiment on the UCI Sonar classification data.

The potential is a regularized squared classification loss for the same
bounded-output two-layer tanh network used in the synthetic neural-network
experiment.  With 60 input features and four hidden units, the trainable
parameter has dimension 248.

The script performs four statistically separated stages:

1. deterministic multistart optimization identifies reference basins;
2. a method-symmetric validation set selects the common cooling energy;
3. two-stage validation tunes UBU and third-order kinetic parameters;
4. fresh paths evaluate basin success and post-quench test accuracy.

Run ``python numerics/run_uci_sonar_experiment.py`` for the paper settings or
add ``--quick`` for a smoke test.  Outputs:

  * numerics/uci_sonar_results.json
  * arxiv/figures/uci_sonar_annealing.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_experiments import COLORS, LABELS, METHODS, make_grid
from run_neural_network_experiment import (
    NetworkConfiguration,
    TwoLayerTanhPotential,
    find_reference_basins,
    local_quench,
    serializable_reference,
    simulate,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "numerics" / "data" / "uci_sonar" / "sonar.all-data"
RESULTS_PATH = ROOT / "numerics" / "uci_sonar_results.json"
FIGURE_PATH = ROOT / "arxiv" / "figures" / "uci_sonar_annealing.png"


def load_sonar_split(
    *, split_seed: int = 151, test_size: int = 48
) -> dict[str, np.ndarray]:
    raw = np.genfromtxt(DATA_PATH, delimiter=",", dtype=str)
    if raw.shape != (208, 61):
        raise RuntimeError(f"unexpected UCI Sonar data shape: {raw.shape}")
    features = raw[:, :60].astype(float)
    targets = np.where(raw[:, 60] == "M", 1.0, -1.0)

    rng = np.random.default_rng(split_seed)
    train_indices: list[int] = []
    test_indices: list[int] = []
    for label in (-1.0, 1.0):
        indices = np.flatnonzero(targets == label)
        rng.shuffle(indices)
        class_test_size = round(test_size * len(indices) / len(targets))
        test_indices.extend(indices[:class_test_size].tolist())
        train_indices.extend(indices[class_test_size:].tolist())
    train = np.asarray(sorted(train_indices))
    test = np.asarray(sorted(test_indices))
    if len(test) != test_size:
        raise RuntimeError("stratified split did not produce the requested size")

    mean = np.mean(features[train], axis=0)
    standard_deviation = np.std(features[train], axis=0)
    standard_deviation[standard_deviation < 1.0e-12] = 1.0
    # Dividing by sqrt(d) keeps a typical feature vector at order-one norm,
    # matching the scale used in the synthetic network experiment.
    scaled = (
        (features - mean)
        / standard_deviation
        / math.sqrt(features.shape[1])
    )
    return {
        "train_features": scaled[train],
        "train_targets": targets[train],
        "test_features": scaled[test],
        "test_targets": targets[test],
        "train_indices": train,
        "test_indices": test,
        "feature_mean": mean,
        "feature_standard_deviation": standard_deviation,
    }


def accuracy(
    potential: TwoLayerTanhPotential,
    theta: np.ndarray,
    features: np.ndarray,
    targets: np.ndarray,
) -> float:
    predictions = potential.predictions(theta, features)
    labels = np.where(predictions >= 0.0, 1.0, -1.0)
    return float(np.mean(labels == targets))


def quench_metrics(
    potential: TwoLayerTanhPotential,
    endpoints: np.ndarray,
    *,
    best_reference: float,
    success_tolerance: float,
    train_features: np.ndarray | None = None,
    train_targets: np.ndarray | None = None,
    test_features: np.ndarray | None = None,
    test_targets: np.ndarray | None = None,
) -> dict:
    values = np.empty(len(endpoints))
    converged = np.zeros(len(endpoints), dtype=bool)
    gradient_norms = np.empty(len(endpoints))
    report_accuracy = train_features is not None
    train_accuracy = np.empty(len(endpoints)) if report_accuracy else None
    test_accuracy = np.empty(len(endpoints)) if report_accuracy else None
    for index, endpoint in enumerate(endpoints):
        point, value, gradient_norm, did_converge = local_quench(
            potential, endpoint, maxiter=800
        )
        values[index] = value
        gradient_norms[index] = gradient_norm
        converged[index] = did_converge
        if report_accuracy:
            assert train_features is not None and train_targets is not None
            assert test_features is not None and test_targets is not None
            train_accuracy[index] = accuracy(
                potential, point, train_features, train_targets
            )
            test_accuracy[index] = accuracy(
                potential, point, test_features, test_targets
            )

    success = values <= best_reference + success_tolerance
    count = len(endpoints)
    result = {
        "quenched_objectives": values.tolist(),
        "n_quench_converged": int(np.sum(converged)),
        "maximum_quench_gradient_norm": float(np.max(gradient_norms)),
        "best_basin_success": float(np.mean(success)),
        "best_basin_success_se": float(
            np.std(success, ddof=1) / math.sqrt(count)
        ),
        "mean_quenched_objective": float(np.mean(values)),
        "mean_quenched_objective_se": float(
            np.std(values, ddof=1) / math.sqrt(count)
        ),
    }
    if report_accuracy:
        assert train_accuracy is not None and test_accuracy is not None
        result.update(
            {
                "train_accuracies": train_accuracy.tolist(),
                "test_accuracies": test_accuracy.tolist(),
                "mean_train_accuracy": float(np.mean(train_accuracy)),
                "mean_train_accuracy_se": float(
                    np.std(train_accuracy, ddof=1) / math.sqrt(count)
                ),
                "mean_test_accuracy": float(np.mean(test_accuracy)),
                "mean_test_accuracy_se": float(
                    np.std(test_accuracy, ddof=1) / math.sqrt(count)
                ),
                "best_basin_mean_test_accuracy": float(
                    np.mean(test_accuracy[success]) if np.any(success) else np.nan
                ),
            }
        )
    return result


def evaluate_candidate(
    method: str,
    potential: TwoLayerTanhPotential,
    reference: dict,
    tau: np.ndarray,
    steps: np.ndarray,
    *,
    n_paths: int,
    energy: float,
    seed: int,
    eta: float,
    lam: float,
    gamma: float,
    success_tolerance: float,
) -> dict:
    result = simulate(
        method,
        potential,
        reference["start_point"],
        tau,
        steps,
        n_paths=n_paths,
        energy=energy,
        temperature_shift=math.e,
        seed=seed,
        eta=eta,
        lam=lam,
        gamma=gamma,
    )
    quenched = quench_metrics(
        potential,
        result["endpoints"],
        best_reference=reference["best_value"],
        success_tolerance=success_tolerance,
    )
    return {
        key: value
        for key, value in quenched.items()
        if key != "quenched_objectives"
    }


def select_temperature(
    potential: TwoLayerTanhPotential,
    reference: dict,
    tau: np.ndarray,
    steps: np.ndarray,
    *,
    energy_grid: tuple[float, ...],
    n_paths: int,
    success_tolerance: float,
) -> dict:
    rows: list[dict] = []
    for energy_id, energy in enumerate(energy_grid):
        scores: dict[str, float] = {}
        for method_id, method in enumerate(METHODS):
            row = evaluate_candidate(
                method,
                potential,
                reference,
                tau,
                steps,
                n_paths=n_paths,
                energy=energy,
                seed=51000 + 10 * energy_id + method_id,
                eta=0.1,
                lam=0.5,
                gamma=4.0,
                success_tolerance=success_tolerance,
            )
            scores[method] = row["best_basin_success"]
        result = {
            "energy": energy,
            "method_success": scores,
            "pooled_success": float(np.mean(list(scores.values()))),
        }
        rows.append(result)
        print("temperature validation", result, flush=True)
    selected = min(rows, key=lambda row: abs(row["pooled_success"] - 0.35))
    return {
        "energy_grid": list(energy_grid),
        "n_paths_per_method": n_paths,
        "selection_rule": "pooled success closest to 0.35",
        "selected_energy": selected["energy"],
        "results": rows,
    }


def tune_parameters(
    potential: TwoLayerTanhPotential,
    reference: dict,
    tau: np.ndarray,
    steps: np.ndarray,
    *,
    energy: float,
    screening_paths: int,
    confirmation_paths: int,
    success_tolerance: float,
) -> dict:
    eta_grid = (0.05, 0.1, 0.25, 0.5, 1.0)
    lambda_grid = (0.25, 0.5, 1.0)
    gamma_grid = (2.0, 4.0, 8.0)

    def key(row: dict) -> tuple[float, float]:
        return row["best_basin_success"], -row["mean_quenched_objective"]

    screening = {"underdamped": [], "third_order": []}
    for eta in eta_grid:
        row = evaluate_candidate(
            "underdamped",
            potential,
            reference,
            tau,
            steps,
            n_paths=screening_paths,
            energy=energy,
            seed=61001,
            eta=eta,
            lam=0.5,
            gamma=4.0,
            success_tolerance=success_tolerance,
        )
        row["eta"] = eta
        screening["underdamped"].append(row)
        print("screen UBU", row, flush=True)
    for lam in lambda_grid:
        for gamma in gamma_grid:
            row = evaluate_candidate(
                "third_order",
                potential,
                reference,
                tau,
                steps,
                n_paths=screening_paths,
                energy=energy,
                seed=61002,
                eta=0.1,
                lam=lam,
                gamma=gamma,
                success_tolerance=success_tolerance,
            )
            row["lambda"] = lam
            row["gamma"] = gamma
            screening["third_order"].append(row)
            print("screen third order", row, flush=True)

    finalists = {
        method: sorted(rows, key=key, reverse=True)[:2]
        for method, rows in screening.items()
    }
    confirmation = {"underdamped": [], "third_order": []}
    for candidate in finalists["underdamped"]:
        row = evaluate_candidate(
            "underdamped",
            potential,
            reference,
            tau,
            steps,
            n_paths=confirmation_paths,
            energy=energy,
            seed=62001,
            eta=candidate["eta"],
            lam=0.5,
            gamma=4.0,
            success_tolerance=success_tolerance,
        )
        row["eta"] = candidate["eta"]
        confirmation["underdamped"].append(row)
        print("confirm UBU", row, flush=True)
    for candidate in finalists["third_order"]:
        row = evaluate_candidate(
            "third_order",
            potential,
            reference,
            tau,
            steps,
            n_paths=confirmation_paths,
            energy=energy,
            seed=62002,
            eta=0.1,
            lam=candidate["lambda"],
            gamma=candidate["gamma"],
            success_tolerance=success_tolerance,
        )
        row["lambda"] = candidate["lambda"]
        row["gamma"] = candidate["gamma"]
        confirmation["third_order"].append(row)
        print("confirm third order", row, flush=True)

    selected_ud = max(confirmation["underdamped"], key=key)
    selected_third = max(confirmation["third_order"], key=key)
    return {
        "screening_paths_per_candidate": screening_paths,
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


def run_evaluation(
    potential: TwoLayerTanhPotential,
    reference: dict,
    split: dict[str, np.ndarray],
    tau: np.ndarray,
    steps: np.ndarray,
    *,
    n_paths: int,
    energy: float,
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
            temperature_shift=math.e,
            seed=seed_base + method_id,
            eta=eta,
            lam=lam,
            gamma=gamma,
        )
        print(f"evaluation {method}: quenching {n_paths} endpoints", flush=True)
        summary = quench_metrics(
            potential,
            simulation.pop("endpoints"),
            best_reference=reference["best_value"],
            success_tolerance=success_tolerance,
            train_features=split["train_features"],
            train_targets=split["train_targets"],
            test_features=split["test_features"],
            test_targets=split["test_targets"],
        )
        results[method] = {"summary": summary, "curves": simulation["curves"]}
        print(
            method,
            {
                "best_basin_success": summary["best_basin_success"],
                "mean_test_accuracy": summary["mean_test_accuracy"],
                "mean_quenched_objective": summary["mean_quenched_objective"],
            },
            flush=True,
        )
    return results


def plot_results(reference: dict, evaluation: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.7, 3.9))
    best = reference["best_value"]
    multistart_values: list[float] = []
    for cluster in reference["clusters"]:
        multistart_values.extend(
            [1.0e3 * (cluster["mean_value"] - best)] * cluster["count"]
        )
    axes[0].hist(multistart_values, bins=12, color="#999999", edgecolor="white")
    axes[0].set_xlabel(r"Local-minimum gap from best found ($\times10^{-3}$)")
    axes[0].set_ylabel("Multistart count")
    axes[0].set_title("UCI Sonar basin structure")
    axes[0].grid(alpha=0.2, axis="y")

    xloc = np.arange(len(METHODS))
    success = [
        evaluation[method]["summary"]["best_basin_success"] for method in METHODS
    ]
    success_se = [
        evaluation[method]["summary"]["best_basin_success_se"] for method in METHODS
    ]
    axes[1].bar(
        xloc,
        success,
        yerr=success_se,
        capsize=4,
        color=[COLORS[method] for method in METHODS],
    )
    axes[1].set_xticks(xloc, ("OD", "UD", "Third"))
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("Best-basin probability")
    axes[1].set_title("Held-out annealing paths")
    axes[1].grid(alpha=0.2, axis="y")

    test_accuracy = [
        evaluation[method]["summary"]["mean_test_accuracy"] for method in METHODS
    ]
    accuracy_se = [
        evaluation[method]["summary"]["mean_test_accuracy_se"] for method in METHODS
    ]
    axes[2].bar(
        xloc,
        test_accuracy,
        yerr=accuracy_se,
        capsize=4,
        color=[COLORS[method] for method in METHODS],
    )
    axes[2].set_xticks(xloc, ("OD", "UD", "Third"))
    axes[2].set_ylim(0.7, 1.0)
    axes[2].set_ylabel("Post-quench test accuracy")
    axes[2].set_title("48 held-out Sonar records")
    axes[2].grid(alpha=0.2, axis="y")

    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small smoke-test run")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="regenerate the figure from numerics/uci_sonar_results.json",
    )
    args = parser.parse_args()
    if args.plot_only:
        payload = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        plot_results(payload["reference_basins"], payload["evaluation"])
        print(f"wrote {FIGURE_PATH}", flush=True)
        return

    split = load_sonar_split()
    config = NetworkConfiguration(
        input_dimension=60,
        hidden_units=4,
        teacher_hidden_units=0,
        sample_size=len(split["train_targets"]),
        output_scale=2.0,
        regularization=1.0e-3,
        data_seed=151,
    )
    potential = TwoLayerTanhPotential(
        config,
        features=split["train_features"],
        targets=split["train_targets"],
    )
    if args.quick:
        n_reference_starts = 24
        n_temperature_paths = 12
        screening_paths = 8
        confirmation_paths = 16
        evaluation_paths = 48
        refinement_paths = 24
    else:
        n_reference_starts = 64
        n_temperature_paths = 24
        screening_paths = 32
        confirmation_paths = 96
        evaluation_paths = 256
        refinement_paths = 64

    h0 = 0.05
    horizon = 20.0
    success_tolerance = 1.0e-4
    tau, steps = make_grid(
        h0=h0, exponent=0.5, t0=math.e, horizon=horizon
    )
    reference = find_reference_basins(
        potential, n_starts=n_reference_starts, seed=21510
    )
    print(
        "reference basins",
        [(row["mean_value"], row["count"]) for row in reference["clusters"]],
        flush=True,
    )
    reference["best_train_accuracy"] = accuracy(
        potential,
        reference["best_point"],
        split["train_features"],
        split["train_targets"],
    )
    reference["best_test_accuracy"] = accuracy(
        potential,
        reference["best_point"],
        split["test_features"],
        split["test_targets"],
    )
    reference["start_train_accuracy"] = accuracy(
        potential,
        reference["start_point"],
        split["train_features"],
        split["train_targets"],
    )
    reference["start_test_accuracy"] = accuracy(
        potential,
        reference["start_point"],
        split["test_features"],
        split["test_targets"],
    )

    temperature_validation = select_temperature(
        potential,
        reference,
        tau,
        steps,
        # The initial grid ended at 0.20 while the pooled success rate was
        # still below the prespecified 0.35 target.  The three larger values
        # keep the same method-symmetric validation rule while avoiding a
        # boundary-selected temperature.
        energy_grid=(0.08, 0.12, 0.16, 0.20, 0.24, 0.30, 0.40),
        n_paths=n_temperature_paths,
        success_tolerance=success_tolerance,
    )
    energy = temperature_validation["selected_energy"]
    tuning = tune_parameters(
        potential,
        reference,
        tau,
        steps,
        energy=energy,
        screening_paths=screening_paths,
        confirmation_paths=confirmation_paths,
        success_tolerance=success_tolerance,
    )
    eta = tuning["selected"]["underdamped"]["eta"]
    lam = tuning["selected"]["third_order"]["lambda"]
    gamma = tuning["selected"]["third_order"]["gamma"]
    print(
        f"selected E={energy}, eta={eta}, lambda={lam}, gamma={gamma}",
        flush=True,
    )

    evaluation = run_evaluation(
        potential,
        reference,
        split,
        tau,
        steps,
        n_paths=evaluation_paths,
        energy=energy,
        seed_base=71000,
        eta=eta,
        lam=lam,
        gamma=gamma,
        success_tolerance=success_tolerance,
    )
    refined_tau, refined_steps = make_grid(
        h0=0.5 * h0, exponent=0.5, t0=math.e, horizon=horizon
    )
    refinement = run_evaluation(
        potential,
        reference,
        split,
        refined_tau,
        refined_steps,
        n_paths=refinement_paths,
        energy=energy,
        seed_base=81000,
        eta=eta,
        lam=lam,
        gamma=gamma,
        success_tolerance=success_tolerance,
    )

    plot_results(reference, evaluation)
    payload = {
        "settings": {
            "quick": args.quick,
            "dataset": {
                "name": "Connectionist Bench (Sonar, Mines vs. Rocks)",
                "uci_id": 151,
                "doi": "10.24432/C5T01Q",
                "license": "CC BY 4.0",
                "n_records": 208,
                "n_features": 60,
                "train_size": len(split["train_targets"]),
                "test_size": len(split["test_targets"]),
                "split_seed": 151,
            },
            "network": {
                "hidden_units": config.hidden_units,
                "parameter_dimension": config.parameter_dimension,
                "output_scale": config.output_scale,
                "regularization": config.regularization,
            },
            "schedule": "epsilon_t = E / log(t + e)",
            "temperature_energy": energy,
            "horizon": horizon,
            "initial_step": h0,
            "step_exponent": 0.5,
            "n_steps": len(steps),
            "kinetic_parameters": {
                "eta": eta,
                "lambda": lam,
                "gamma": gamma,
            },
            "success_tolerance": success_tolerance,
        },
        "split": {
            "train_indices": split["train_indices"].tolist(),
            "test_indices": split["test_indices"].tolist(),
        },
        "temperature_validation": temperature_validation,
        "kinetic_parameter_tuning": tuning,
        "reference_basins": serializable_reference(reference),
        "evaluation": evaluation,
        "step_refinement": refinement,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {RESULTS_PATH}", flush=True)
    print(f"wrote {FIGURE_PATH}", flush=True)


if __name__ == "__main__":
    main()
