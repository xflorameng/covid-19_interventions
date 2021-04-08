import numpy as np

from models import monte_carlo


DIR_NAME_TAG = '_subsidy'
NUM_HOUSEHOLDS = {'poor': int(2.04e5), 'rich': int(3.4e4)}
MAX_HOUSEHOLD_SIZES = {'poor': 5, 'rich': 3}
NUM_TRIALS = 100

monte_carlo(subsidy=np.arange(0, 0.55, 0.05),
            num_households=NUM_HOUSEHOLDS, max_household_sizes=MAX_HOUSEHOLD_SIZES, num_trials=NUM_TRIALS)

# Robustness test of same-age households
monte_carlo(same_age_household=True, subsidy=np.arange(0, 0.55, 0.05),
            num_households=NUM_HOUSEHOLDS, max_household_sizes=MAX_HOUSEHOLD_SIZES, num_trials=NUM_TRIALS)
