# COVID-19 Interventions
Mechanistic simulation of health and economic impacts of various interventions during the COVID-19 pandemic

# Overview
The agent-based model incorporates key elements including socioeconomic status, age-dependent risks, household transmission, asymptomatic transmission, and hospital capacity.

# System Requirements
## Hardware Requirements
A standard computer with enough RAM to support the in-memory operations.
## Software Requirements
The code has been tested on macOS Big Sur (Version 11.2.3).
## Python Dependencies
Python 3.7 with the following packages
```
numpy (1.18.2)
pandas (1.0.3)
tqdm (4.45.0)
networkx (2.4)
statsmodels (0.12.1)
matplotlib (3.2.1)
seaborn (0.10.0)
sklearn (0.0)
dtreeviz (1.1.4)
pytest (5.4.1)
```

# Installation Guide
Please download the entire repository and set up a virtual environment of Python 3.7 with packages as detailed above. Installation typically takes a few minutes on a standard computer.

# Demo and Instructions
Example scripts have been provided in the `src` directory.
Specifically, `monte_carlo_simulation_lockdown.py` conducts Monte Carlo experiments to study the effects of lockdown at various levels;
`monte_carlo_simulation_testing.py` explores testing;
`monte_carlo_simulation_subsidy.py` studies subsidization;
`monte_carlo_simulation_greedy_subsidy_with_budget.py` considers greedy subsidization under a budget constraint;
`monte_carlo_simulation_overcrowding.py` investigates household overcrowding.
Please use `monte_carlo_plot.py` to visualize results of the Monte Carlo experiments.
Additionally, `single_simulation.py` gives an example that conducts in-depth analysis of one single simulation.
The expected outputs are in the `results` directory. For the example configurations, the run time of one single simulation ranges from a few minutes to an hour on a standard computer.

We also provide scripts in the `analysis` directory that conduct analysis of US data to corroborate simulation results. Specifically, `feature_importance.py` trains a decision tree and computes feature importance for the task of estimating regional COVID-19 death rates. `lockdown_tradeoffs.py` illustrates the health and economic trade-off dependent on the lockdown level that is unique to socioeconomically disadvantaged populations. `overcrowding.py` investigates the effects of household overcrowding on the regional COVID-19 death rate. The results are all saved in the `outputs` directory. The run time is a few seconds on a standard computer for `feature_importance.py` and `lockdown_tradeoffs.py`. The script `overcrowding.py` takes a few minutes to run on a standard computer.
