from models import monte_carlo_multi_plots, monte_carlo_multi_controls, monte_carlo_multi_plots_approx


csv_path = 'monte_carlo_data_lockdown_level/'
# csv_path = 'monte_carlo_data_lockdown_level_same_age/'
x, x_label, x_method = ['worker_staying_at_home_prob'], 'Lockdown level', None

# csv_path = 'monte_carlo_data_testing_rate/'
# csv_path = 'monte_carlo_data_testing_rate_same_age/'
# x, x_label, x_method = ['viral_test_fraction_all'], 'Testing rate', None

# csv_path = 'monte_carlo_data_subsidy/'
# csv_path = 'monte_carlo_data_subsidy_same_age/'
# x, x_label, x_method = ['subsidy'], 'Subsidy', None

# csv_path = 'monte_carlo_data_household_size/'
# csv_path = 'monte_carlo_data_household_size_same_age/'
# x, x_label, x_method = ['max_household_sizes', 'poor'], 'Maximum poor household size', None

ys = [['virus_related_deaths'], ['deaths_of_despair'], ['total_deaths']]
y_labels = ['COVID-19', 'Despair', 'Total']
y_methods = ['max', 'max', 'max']

legend_kwargs = {'fontsize': 12}
ordered_legends = True
y_axis_label = r'Deaths per $10^5$ people'
y_scale = 1e5
ylim = None
monte_carlo_multi_plots(x, x_label, x_method, ys, y_labels, y_methods, y_axis_label,
                        y_scale=y_scale, ylim=ylim, std=True, by_household_type=True, drop_after_time=None,
                        figsize=(4.5, 3.5), legend=True, ordered_legends=ordered_legends, legend_kwargs=legend_kwargs,
                        save=True, timestamp=True, csv_path=csv_path)


# Greedy subsidy with budget
csv_path = 'monte_carlo_data_greedy_subsidy/'
x, x_label = ['worker_staying_at_home_prob'], 'Lockdown level'
y, y_label, y_method = ['total_deaths'], r'Total deaths per $10^5$ people', 'max'
control, control_label = ['subsidy_population_factor'], 'Budget level'

legend_kwargs = {'fontsize': 12}
y_scale = 1e5
ylim = None
monte_carlo_multi_controls(x, x_label, y, y_label, y_method, control, control_label,
                           y_scale=y_scale, ylim=ylim, std=True, by_household_type=True, drop_after_time=None,
                           figsize=(4.5, 3.5), legend_kwargs=legend_kwargs, save=True, timestamp=True, csv_path=csv_path)


# Mean-field approximation
drop_after_time = 44
csv_path = 'monte_carlo_data_household_size/'
x, x_label, x_method = ['max_household_sizes', 'poor'], 'Maximum poor household size', None

ys = [['infections_cumsum']]
y_labels = ['Simulation']
y_methods = ['max']

legend_kwargs = {'fontsize': 12}
ylim = None
monte_carlo_multi_plots_approx(x, x_label, x_method, ys, y_labels, y_methods,
                               ylim=ylim, std=True, by_household_type=True, drop_after_time=drop_after_time,
                               figsize=(4.5, 3.5), legend_kwargs=legend_kwargs,
                               save=True, timestamp=True, csv_path=csv_path)
