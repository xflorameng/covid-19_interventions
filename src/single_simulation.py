import math

import numpy as np
from tqdm import tqdm
from networkx.generators.random_graphs import watts_strogatz_graph, fast_gnp_random_graph

from src.models import Simulation


# Policy parameters
OCCUPATIONS_STAYING_AT_HOME = 'retiree'
# OCCUPATIONS_STAYING_AT_HOME = ''
PARTIAL_OPENING = True
WORKER_STAYING_AT_HOME_PROB = .4
STAYING_AT_HOME_OUTPUT_REMAINING_SCALE = 1.0
VIRAL_TEST_FRACTION_ALL = 1.45e-3
VIRAL_TEST_FRACTION_POOR = None
# VIRAL_TEST_FRACTION_POOR = 1.55e-3
CONTACT_TRACING_EFFICACY = .7
SUBSIDY = .1
# SUBSIDY = None

# Network parameters
HOUSEHOLD_NETWORK = True
SAME_AGE_HOUSEHOLD = False
NUM_HOUSEHOLDS = {'poor': int(6e3), 'rich': int(1e3)}
MAX_HOUSEHOLD_SIZES = {'poor': 5, 'rich': 3}
# NUM_HOUSEHOLDS = int(2.67e5)
# MAX_HOUSEHOLD_SIZES = 4
ECON_NETWORK = 'watts strogatz'
# ECON_NETWORK = 'fast gnp'
AVERAGE_WORKER_DEGREE = 20
REWIRING_PROB = .5  # For watts-strogatz random graph
VULNERABLE_GROUP = True
VULNERABLE_POPULATION_FRACTION = .01
VULNERABILITY = .05

# Epidemiological parameters
INITIAL_INFECTION_FRACTION = 1e-3
HOUSEHOLD_TRANSMISSION_PROB = .25
OTHER_TRANSMISSION_PROB = HOUSEHOLD_TRANSMISSION_PROB / 50

# Other parameters
DESPAIR_PROB_FACTOR = 5.5e-5
INACTIVE_PROB_FACTOR = 1e-2
HOSPITAL_CAPACITY = 2.5e-3  # Calibrated to MA
UNDERTREATMENT_EFFECT = .5

# Timing of lockdown without reopening
TIME_STEPS_PRE_LOCKDOWN = 5
TIME_STEPS_POST_LOCKDOWN_PRE_REOPENING = 175
TIME_STEPS_POST_REOPENING = 0

# # No lockdown
# TIME_STEPS_PRE_LOCKDOWN = 180
# TIME_STEPS_POST_LOCKDOWN_PRE_REOPENING = 0
# TIME_STEPS_POST_REOPENING = 0

# # Timing of lockdown and reopening
# TIME_STEPS_PRE_LOCKDOWN = 5
# TIME_STEPS_POST_LOCKDOWN_PRE_REOPENING = 30
# TIME_STEPS_POST_REOPENING = 145


print('--------------------Initializing the simulation------------------------')
sim = Simulation()
sim.calc_output_measures(AVERAGE_WORKER_DEGREE)
print('Done')

print('--------------------Generating networks--------------------------------')
if HOUSEHOLD_NETWORK:
    if isinstance(NUM_HOUSEHOLDS, int):
        sim.household_network(NUM_HOUSEHOLDS, MAX_HOUSEHOLD_SIZES, None, 1.0, SAME_AGE_HOUSEHOLD)
    else:
        sim.household_network(NUM_HOUSEHOLDS['poor'], MAX_HOUSEHOLD_SIZES['poor'], 'poor', 1.0, SAME_AGE_HOUSEHOLD)
        sim.household_network(NUM_HOUSEHOLDS['rich'], MAX_HOUSEHOLD_SIZES['rich'], 'rich', 1.0, SAME_AGE_HOUSEHOLD)
        sim.calc_rich_to_poor_output_ratio()
if not sim.household_grouping:
    average_num_nodes = int(NUM_HOUSEHOLDS * (1 + MAX_HOUSEHOLD_SIZES) / 2)
else:
    average_num_nodes = int((NUM_HOUSEHOLDS['poor'] * (1 + MAX_HOUSEHOLD_SIZES['poor'])
                             + NUM_HOUSEHOLDS['rich'] * (1 + MAX_HOUSEHOLD_SIZES['rich'])) / 2)
# For a list of network models, see:
# https://networkx.github.io/documentation/stable/reference/generators.html
if ECON_NETWORK == 'watts strogatz':
    sim.econ_network(watts_strogatz_graph, [average_num_nodes, AVERAGE_WORKER_DEGREE, REWIRING_PROB])
elif ECON_NETWORK == 'fast gnp':
    worker_edge_prob = AVERAGE_WORKER_DEGREE / average_num_nodes
    sim.econ_network(fast_gnp_random_graph, [average_num_nodes, worker_edge_prob])
sim.populate_connections()  # Put this line after all networks are generated
if VULNERABLE_GROUP:  # Put this line after all networks are generated
    sim.add_vulnerable_group(VULNERABLE_POPULATION_FRACTION, VULNERABILITY)
print('Done')

print('--------------------Seeding initial infections-------------------------')
num_initial_infections = math.ceil(INITIAL_INFECTION_FRACTION * sim.num_nodes)
initial_infections = np.random.choice(sim.num_nodes, num_initial_infections, replace=False)
sim.seed_simulation(initial_infections, record_stats=True)
print('Done')

print('--------------------Calculating testing capacity-----------------------')
if VIRAL_TEST_FRACTION_POOR is None:
    num_viral_tests = int(VIRAL_TEST_FRACTION_ALL * sim.num_nodes)
else:
    rich_fraction = sim.num_nodes_rich / sim.num_nodes
    viral_test_fraction_rich = (VIRAL_TEST_FRACTION_ALL - (1 - rich_fraction)
                                * VIRAL_TEST_FRACTION_POOR) / rich_fraction
    assert 0 <= viral_test_fraction_rich <= 1, 'Testing rate for the rich must be between 0 and 1'
    num_viral_tests = {'poor': int(VIRAL_TEST_FRACTION_POOR * sim.num_nodes_poor),
                       'rich': int(viral_test_fraction_rich * sim.num_nodes_rich)}
print('Done')

print('--------------------Configuring simulation-----------------------------')
def simulation_step():
    sim.simulation_step(HOUSEHOLD_TRANSMISSION_PROB, OTHER_TRANSMISSION_PROB, num_viral_tests, CONTACT_TRACING_EFFICACY,
                        HOSPITAL_CAPACITY, UNDERTREATMENT_EFFECT, DESPAIR_PROB_FACTOR, SUBSIDY, INACTIVE_PROB_FACTOR,
                        record_stats=True)
print('Done')

print('--------------------Pre-lockdown simulation begins---------------------')
for _ in tqdm(range(TIME_STEPS_PRE_LOCKDOWN)):
    simulation_step()
print('Done')

if TIME_STEPS_POST_LOCKDOWN_PRE_REOPENING > 0:
    print('--------------------Setting lockdown-----------------------------------')
    if OCCUPATIONS_STAYING_AT_HOME != '':
        sim.stay_at_home_by_occupation_policy(OCCUPATIONS_STAYING_AT_HOME.split(','),
                                              STAYING_AT_HOME_OUTPUT_REMAINING_SCALE)
    if PARTIAL_OPENING:
        sim.partial_opening_policy(WORKER_STAYING_AT_HOME_PROB, STAYING_AT_HOME_OUTPUT_REMAINING_SCALE)
    print('Done')

    print('--------------------Post-lockdown simulation begins--------------------')
    for _ in tqdm(range(TIME_STEPS_POST_LOCKDOWN_PRE_REOPENING)):
        simulation_step()
    print('Done')

    if TIME_STEPS_POST_REOPENING > 0:
        print('--------------------Reopening--------------------')
        sim.reopen_policy()
        print('Done')

        print('--------------------Post-reopening simulation begins--------------------')
        for _ in tqdm(range(TIME_STEPS_POST_REOPENING)):
            simulation_step()
        print('Done')

print('--------------------Saving statistics to CSV---------------------------')
sim.save_to_csv(timestamp=False, path='../results/single_simulation/')
print('Done')

print('--------------------Plotting results-----------------------------------')
sim.plot_age_dist(save=True, timestamp=False, path='../results/single_simulation/')
sim.plot_p_despair(AVERAGE_WORKER_DEGREE, DESPAIR_PROB_FACTOR, x_label='Output loss',
                   figsize=(4.5, 3.5), save=True, timestamp=False, path='../results/single_simulation/')

y_strs_health = ['isolation_count', 'infections', 'recoveries', 'hospitalizations', 'ICU_count',
                 'viral_deaths', 'undertreated_deaths', 'deaths_of_despair']
y_labels_health = ['Isolation', 'Infections', 'Recoveries', 'Hospitalizations', 'ICU',
                   'Deaths from virus', 'Deaths from undertreatment', 'Deaths of despair']

y_strs_econ = ['stay_at_home_count', 'active_count', 'total_output', 'total_subsidy']
y_labels_econ = ['Stay at home', 'Active', 'Total output', 'Total subsidy']

if TIME_STEPS_POST_LOCKDOWN_PRE_REOPENING == 0:
    vlines = None
    vline_labels = None
elif TIME_STEPS_POST_REOPENING == 0:
    vlines = (TIME_STEPS_PRE_LOCKDOWN,)
    vline_labels = ('Lockdown date',)
else:
    time_steps_pre_reopening = TIME_STEPS_PRE_LOCKDOWN + TIME_STEPS_POST_LOCKDOWN_PRE_REOPENING
    vlines = (TIME_STEPS_PRE_LOCKDOWN, time_steps_pre_reopening),
    vline_labels = ('Lockdown date', 'Reopening date')

if not sim.household_grouping:
    title = 'No household grouping'
    sim.plot_all_time_series(title, normalized=True, household_type=None, y_lims=(-.001, .011, -.05, 1.05),
                             vlines=vlines, vline_labels=vline_labels,
                             save=True, timestamp=False, path='../results/single_simulation/')
    sim.plot_econ_time_series(title, normalized=False, household_type=None, vlines=vlines, vline_labels=vline_labels,
                              save=True, timestamp=False, path='../results/single_simulation/')
    sim.plot_time_series(y_strs_health, y_labels_health, title, normalized=True, household_type=None,
                         vlines=vlines, vline_labels=vline_labels, figsize=(5, 3),
                         save=True, filename_tag='health_measures_', timestamp=False,
                         path='../results/single_simulation/')
    sim.plot_time_series(y_strs_econ, y_labels_econ, title, normalized=True, household_type=None,
                         vlines=vlines, vline_labels=vline_labels,
                         figsize=(5, 3), save=True, filename_tag='econ_measures_', timestamp=False,
                         path='../results/single_simulation/')
else:
    titles = ['All', 'Poor', 'Rich']
    household_types = [None, 'poor', 'rich']
    for i in range(len(titles)):
        sim.plot_all_time_series(titles[i], normalized=True, household_type=household_types[i],
                                 y_lims=(-.001, .011, -.05, 1.05), vlines=vlines, vline_labels=vline_labels,
                                 save=True, timestamp=False, path='../results/single_simulation/')
        sim.plot_econ_time_series(titles[i], normalized=False, household_type=household_types[i],
                                  vlines=vlines, vline_labels=vline_labels, save=True, timestamp=False,
                                  path='../results/single_simulation/')
        sim.plot_time_series(y_strs_health, y_labels_health, titles[i],
                             normalized=True, household_type=household_types[i],
                             vlines=vlines, vline_labels=vline_labels,
                             figsize=(5, 3), save=True, filename_tag='health_measures', timestamp=False,
                             path='../results/single_simulation/')
        sim.plot_time_series(y_strs_econ, y_labels_econ, titles[i],
                             normalized=True, household_type=household_types[i],
                             vlines=vlines, vline_labels=vline_labels,
                             figsize=(5, 3), save=True, filename_tag='econ_measures', timestamp=False,
                             path='../results/single_simulation/')
print('Done')
