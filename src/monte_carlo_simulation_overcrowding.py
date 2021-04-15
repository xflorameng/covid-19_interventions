import os
import warnings

import numpy as np

from models import monte_carlo


DIR_NAME_TAG = '_household_size'
MAX_HOUSEHOLD_SIZE_POOR = np.arange(3, 9, 1)
MAX_HOUSEHOLD_SIZE_RICH = 3
NUM_HOUSEHOLDS_RICH = int(3.4e4)
NUM_TRIALS = 100

REWIRING_PROB = .5

TIME_STEPS_PRE_LOCKDOWN = 0
TIME_STEPS_POST_LOCKDOWN_PRE_REOPENING = 60

dir_name = '../results/monte_carlo_data' + DIR_NAME_TAG + '/'
if os.path.exists(dir_name):
    dir_name_former = f'{dir_name[:-1]}_former/'
    os.rename(dir_name, dir_name_former)
    warnings.warn(f'The following directory already exists: {dir_name}\n'
                  f'It has been renamed to: {dir_name_former}')
os.mkdir(dir_name)
os.chdir(dir_name)
for i in MAX_HOUSEHOLD_SIZE_POOR:
    MAX_HOUSEHOLD_SIZES = {'poor': i, 'rich': MAX_HOUSEHOLD_SIZE_RICH}
    num_households_poor = int((1 + MAX_HOUSEHOLD_SIZE_RICH) * NUM_HOUSEHOLDS_RICH * 9 / (1 + i))
    NUM_HOUSEHOLDS = {'poor': num_households_poor, 'rich': NUM_HOUSEHOLDS_RICH}
    monte_carlo(create_dir=False,
                num_households=NUM_HOUSEHOLDS, max_household_sizes=MAX_HOUSEHOLD_SIZES,
                time_steps_pre_lockdown=TIME_STEPS_PRE_LOCKDOWN,
                time_steps_post_lockdown_pre_reopening=TIME_STEPS_POST_LOCKDOWN_PRE_REOPENING,
                rewiring_prob=REWIRING_PROB, num_trials=NUM_TRIALS,
                dir_name_tag=DIR_NAME_TAG, timestamp=False)

# Robustness test of same-age households
dir_name = '../results/monte_carlo_data' + DIR_NAME_TAG + '_same_age/'
os.chdir('../')
if os.path.exists(dir_name):
    dir_name_former = f'{dir_name[:-1]}_former/'
    os.rename(dir_name, dir_name_former)
    warnings.warn(f'The following directory already exists: {dir_name}\n'
                  f'It has been renamed to: {dir_name_former}')
os.mkdir(dir_name)
os.chdir(dir_name)
for i in MAX_HOUSEHOLD_SIZE_POOR:
    MAX_HOUSEHOLD_SIZES = {'poor': i, 'rich': MAX_HOUSEHOLD_SIZE_RICH}
    num_households_poor = int((1 + MAX_HOUSEHOLD_SIZE_RICH) * NUM_HOUSEHOLDS_RICH * 9 / (1 + i))
    NUM_HOUSEHOLDS = {'poor': num_households_poor, 'rich': NUM_HOUSEHOLDS_RICH}
    monte_carlo(create_dir=False,
                num_households=NUM_HOUSEHOLDS, max_household_sizes=MAX_HOUSEHOLD_SIZES,
                time_steps_pre_lockdown=TIME_STEPS_PRE_LOCKDOWN,
                time_steps_post_lockdown_pre_reopening=TIME_STEPS_POST_LOCKDOWN_PRE_REOPENING,
                rewiring_prob=REWIRING_PROB, num_trials=NUM_TRIALS,
                dir_name_tag=DIR_NAME_TAG+'_same_age', timestamp=False)
