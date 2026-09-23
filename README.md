# Simulated Annealing with Third-Order Langevin Dynamics

Numerical code for the simulated annealing experiments comparing overdamped,
underdamped, and third-order Langevin dynamics.

## Requirements

Python 3.10 or later is recommended. Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Experiments

Run the paper settings from the project root:

```bash
python numerics/run_experiments.py
python numerics/run_neural_network_experiment.py
python numerics/run_uci_sonar_experiment.py
```

Add `--quick` to any command for a smaller smoke run. The scripts write result
JSON files under `numerics/` and figures under `figures/` or `arxiv/figures/`.
The checked-in JSON files contain the reported experiment results.

The UCI Sonar experiment expects the dataset at
`numerics/data/uci_sonar/sonar.all-data`. Obtain it from the
[UCI Machine Learning Repository](https://doi.org/10.24432/C5T01Q) and place it
at that path. The dataset is identified in the results as CC BY 4.0.
