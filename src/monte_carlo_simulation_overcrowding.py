import numpy as np

from src.models import monte_carlo


FILENAME_TAG = '_household_size'
MAX_HOUSEHOLD_SIZE_POOR = np.arange(3, 9, 1)
MAX_HOUSEHOLD_SIZE_RICH = 3
NUM_HOUSEHOLDS_RICH = int(3.4e4)
NUM_TRIALS = 100

REWIRING_PROB = .5

TIME_STEPS_PRE_LOCKDOWN = 0
TIME_STEPS_POST_LOCKDOWN_PRE_REOPENING = 60

for i in MAX_HOUSEHOLD_SIZE_POOR:
    MAX_HOUSEHOLD_SIZES = {'poor': i, 'rich': MAX_HOUSEHOLD_SIZE_RICH}
    num_households_poor = int((1 + MAX_HOUSEHOLD_SIZE_RICH) * NUM_HOUSEHOLDS_RICH * 9 / (1 + i))
    NUM_HOUSEHOLDS = {'poor': num_households_poor, 'rich': NUM_HOUSEHOLDS_RICH}
    monte_carlo(create_dir=False,
                num_households=NUM_HOUSEHOLDS, max_household_sizes=MAX_HOUSEHOLD_SIZES,
                time_steps_pre_lockdown=TIME_STEPS_PRE_LOCKDOWN,
                time_steps_post_lockdown_pre_reopening=TIME_STEPS_POST_LOCKDOWN_PRE_REOPENING,
                rewiring_prob=REWIRING_PROB,
                num_trials=NUM_TRIALS, filename_tag=FILENAME_TAG)

# Robustness test of same-age households
for i in MAX_HOUSEHOLD_SIZE_POOR:
    MAX_HOUSEHOLD_SIZES = {'poor': i, 'rich': MAX_HOUSEHOLD_SIZE_RICH}
    num_households_poor = int((1 + MAX_HOUSEHOLD_SIZE_RICH) * NUM_HOUSEHOLDS_RICH * 9 / (1 + i))
    NUM_HOUSEHOLDS = {'poor': num_households_poor, 'rich': NUM_HOUSEHOLDS_RICH}
    monte_carlo(create_dir=False,
                num_households=NUM_HOUSEHOLDS, max_household_sizes=MAX_HOUSEHOLD_SIZES,
                time_steps_pre_lockdown=TIME_STEPS_PRE_LOCKDOWN,
                time_steps_post_lockdown_pre_reopening=TIME_STEPS_POST_LOCKDOWN_PRE_REOPENING,
                rewiring_prob=REWIRING_PROB,
                num_trials=NUM_TRIALS, filename_tag=FILENAME_TAG)
