import numpy as np

from models import monte_carlo


DIR_NAME_TAG = '_testing_rate'
NUM_HOUSEHOLDS = {'poor': int(2.04e5), 'rich': int(3.4e4)}
MAX_HOUSEHOLD_SIZES = {'poor': 5, 'rich': 3}
NUM_TRIALS = 100

monte_carlo(viral_test_fraction_all=np.around(np.arange(0, .11, 1e-2), decimals=3),
            num_households=NUM_HOUSEHOLDS, max_household_sizes=MAX_HOUSEHOLD_SIZES,
            time_steps_pre_lockdown=180, time_steps_post_lockdown_pre_reopening=0, subsidy=0, num_trials=NUM_TRIALS,
            dir_name_tag=DIR_NAME_TAG, timestamp=False, path='../results/')

# Robustness test of same-age households
monte_carlo(same_age_household=True, viral_test_fraction_all=np.around(np.arange(0, .11, 1e-2), decimals=3),
            num_households=NUM_HOUSEHOLDS, max_household_sizes=MAX_HOUSEHOLD_SIZES,
            time_steps_pre_lockdown=180, time_steps_post_lockdown_pre_reopening=0, subsidy=0, num_trials=NUM_TRIALS,
            dir_name_tag=DIR_NAME_TAG+'_same_age', timestamp=False, path='../results/')
