from src.models import monte_carlo_multi_plots


csv_path = '../results/stats/testing_tracing_no_intervention/'
x, x_label, x_method = 'viral_test_fraction_all', 'Testing rate', None

# csv_path = '../results/stats/viral_test_fraction_poor/'
# x, x_label, x_method = 'viral_test_fraction_poor', 'Testing rate (poor)', None

# csv_path = '../results/stats/subsidy/'
# x, x_label, x_method = 'subsidy', 'Subsidy', None

# csv_path = '../results/stats/worker_staying_at_home_prob/'
# x, x_label, x_method = 'worker_staying_at_home_prob', 'Lockdown level', None

ys = ['total_deaths', 'deaths_of_despair', 'virus_related_deaths']
y_labels = ['Total fatalities', 'Deaths of despair', 'Fatalities due to virus & undertreatment']
y_methods = ['max', 'max', 'max']

# ys = ['inactive_count', 'inactive_count', 'loss_in_output', 'loss_in_output', 'total_subsidy', 'total_subsidy']
# y_labels = ['Max frac of economically inactive', 'Average frac of economically inactive',
#             'Max loss in output', 'Average loss in output', 'Max total subsidy', 'Average total subsidy']
# y_methods = ['max', 'average', 'max', 'average', 'max', 'average']

# ys = ['total_deaths', 'deaths_of_despair', 'virus_related_deaths',
#       'inactive_count', 'loss_in_output', 'total_subsidy']
# y_labels = ['Total fatalities', 'Deaths of despair', 'Fatalities due to virus & undertreatment',
#             'Average frac of economically inactive', 'Average loss in output', 'Average total subsidy']
# y_methods = ['max', 'max', 'max', 'average', 'average', 'average']

# ys = ['infection_hospitalization_ratio', 'hospitalization_fatality_ratio', 'infection_fatality_ratio']
# y_labels = ['Infection hospitalization ratio', 'Hospitalization fatality ratio', 'Infection fatality ratio']
# y_methods = ['max', 'max', 'max']

legend_kwargs = None
monte_carlo_multi_plots(x, x_label, x_method, ys, y_labels, y_methods,
                        std=True, by_household_type=False, drop_after_time=None,
                        legend_kwargs=legend_kwargs, save=True, timestamp=True, csv_path=csv_path)

legend_kwargs = {'loc': 'center left', 'bbox_to_anchor': (1, .5)}
monte_carlo_multi_plots(x, x_label, x_method, ys, y_labels, y_methods,
                        std=False, by_household_type=True, drop_after_time=None,
                        legend_kwargs=legend_kwargs, save=True, timestamp=True, csv_path=csv_path)
