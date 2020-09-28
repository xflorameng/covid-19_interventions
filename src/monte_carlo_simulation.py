import numpy as np

from src.models import monte_carlo

NUM_HOUSEHOLDS = {'poor': int(3e6), 'rich': int(5e5)}
MAX_HOUSEHOLD_SIZES = {'poor': 5, 'rich': 3}
NUM_TRIALS = 100

# Effect of testing & tracing without intervention
monte_carlo(viral_test_fraction_all=np.around(np.arange(0, .11, 1e-2), decimals=3),
            num_households=NUM_HOUSEHOLDS, max_household_sizes=MAX_HOUSEHOLD_SIZES,
            time_steps_pre_lockdown=180, time_steps_post_lockdown_pre_reopening=0, subsidy=0, num_trials=NUM_TRIALS)

# Differential testing
monte_carlo(viral_test_fraction_poor=np.arange(1.45e-3, 1.55e-3, 1e-5),
            num_households=NUM_HOUSEHOLDS, max_household_sizes=MAX_HOUSEHOLD_SIZES, num_trials=NUM_TRIALS)

# Subsidy
monte_carlo(subsidy=np.arange(0, 0.55, 0.05),
            num_households=NUM_HOUSEHOLDS, max_household_sizes=MAX_HOUSEHOLD_SIZES, num_trials=NUM_TRIALS)

# Lockdown level
monte_carlo(worker_staying_at_home_prob=np.around(np.arange(0, 1.1, 0.1), decimals=3),
            num_households=NUM_HOUSEHOLDS, max_household_sizes=MAX_HOUSEHOLD_SIZES, num_trials=NUM_TRIALS)
