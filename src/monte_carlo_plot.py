from models import monte_carlo_multi_plots, monte_carlo_multi_controls, monte_carlo_multi_plots_approx


filename_tag_list = ['lockdown_level', 'lockdown_level_same_age',
                     'testing_rate', 'testing_rate_same_age',
                     'subsidy', 'subsidy_same_age',
                     'household_size', 'household_size_same_age'
                     ]
x_spec_list = [(['worker_staying_at_home_prob'], 'Lockdown level', None),
               (['viral_test_fraction_all'], 'Testing rate', None),
               (['subsidy'], 'Subsidy', None),
               (['max_household_sizes', 'poor'], 'Maximum poor household size', None)
               ]
x_spec_list = [t for t in x_spec_list for _ in range(2)]
x_specs = zip(filename_tag_list, x_spec_list)
ys = [['virus_related_deaths'], ['deaths_of_despair'], ['total_deaths']]
y_labels = ['COVID-19', 'Despair', 'Total']
y_methods = ['max', 'max', 'max']
legend_kwargs = {'fontsize': 12}
y_axis_label = r'Deaths per $10^5$ people'
y_scale = 1e5
ylim = None

for filename_tag, (x, x_label, x_method) in x_specs:
    csv_path = f'../results/monte_carlo_data_{filename_tag}/'
    monte_carlo_multi_plots(x, x_label, x_method, ys, y_labels, y_methods, y_axis_label,
                            y_scale=y_scale, ylim=ylim, std=True, by_household_type=True, drop_after_time=None,
                            figsize=(4.5, 3.5), legend=True, legend_kwargs=legend_kwargs,
                            save=True, filename_tag=f'_{filename_tag}',
                            plot_path='../results/plots/', csv_path=csv_path)

# Greedy subsidy with budget
filename_tag = 'greedy_subsidy'
csv_path = f'../results/monte_carlo_data_{filename_tag}/'
x, x_label = ['worker_staying_at_home_prob'], 'Lockdown level'
y, y_label, y_method = ['total_deaths'], r'Total deaths per $10^5$ people', 'max'
control, control_label = ['subsidy_population_factor'], 'Budget level'

legend_kwargs = {'fontsize': 12}
y_scale = 1e5
ylim = None
monte_carlo_multi_controls(x, x_label, y, y_label, y_method, control, control_label,
                           y_scale=y_scale, ylim=ylim, std=True, by_household_type=True, drop_after_time=None,
                           figsize=(4.5, 3.5), legend_kwargs=legend_kwargs,
                           save=True, filename_tag=f'_{filename_tag}', plot_path='../results/plots/', csv_path=csv_path)


# Mean-field approximation
drop_after_time = 44
filename_tag = 'household_size'
csv_path = f'../results/monte_carlo_data_{filename_tag}/'
x, x_label, x_method = ['max_household_sizes', 'poor'], 'Maximum poor household size', None

ys = [['infections_cumsum']]
y_labels = ['Simulation']
y_methods = ['max']

legend_kwargs = {'fontsize': 12}
ylim = None
monte_carlo_multi_plots_approx(x, x_label, x_method, ys, y_labels, y_methods,
                               ylim=ylim, std=True, by_household_type=True, drop_after_time=drop_after_time,
                               figsize=(4.5, 3.5), legend_kwargs=legend_kwargs,
                               save=True, filename_tag=f'_{filename_tag}_mfa',
                               plot_path='../results/plots/', csv_path=csv_path)
