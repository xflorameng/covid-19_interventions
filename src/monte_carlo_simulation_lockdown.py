import numpy as np

from models import monte_carlo


DIR_NAME_TAG = '_lockdown_level'
NUM_HOUSEHOLDS = {'poor': int(2.04e5), 'rich': int(3.4e4)}
MAX_HOUSEHOLD_SIZES = {'poor': 5, 'rich': 3}
NUM_TRIALS = 100

monte_carlo(worker_staying_at_home_prob=np.around(np.arange(0, 1.1, 0.1), decimals=3),
            num_households=NUM_HOUSEHOLDS, max_household_sizes=MAX_HOUSEHOLD_SIZES, num_trials=NUM_TRIALS,
            dir_name_tag=DIR_NAME_TAG, timestamp=False, path='../results/')

# Robustness test of same-age households
monte_carlo(same_age_household=True, worker_staying_at_home_prob=np.around(np.arange(0, 1.1, 0.1), decimals=3),
            num_households=NUM_HOUSEHOLDS, max_household_sizes=MAX_HOUSEHOLD_SIZES, num_trials=NUM_TRIALS,
            dir_name_tag=DIR_NAME_TAG+'_same_age', timestamp=False, path='../results/')
