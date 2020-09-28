from src.models import monte_carlo_plot_cbar


csv_path = '../results/stats/testing_tracing_no_intervention/'
color_by, color_by_label = 'viral_test_fraction_all', 'Testing rate'

# csv_path = '../results/stats/viral_test_fraction_poor/'
# color_by, color_by_label = 'viral_test_fraction_poor', 'Testing rate (poor)'

# csv_path = '../results/stats/subsidy/'
# color_by, color_by_label = 'subsidy', 'Subsidy'

# csv_path = '../results/stats/worker_staying_at_home_prob/'
# color_by, color_by_label = 'worker_staying_at_home_prob', 'Lockdown level'

# x, x_label, x_method = 'total_deaths', 'Fatalities', 'max'
x, x_label, x_method = 'deaths_of_despair', 'Deaths of despair', 'max'
# x, x_label, x_method = 'virus_related_deaths', 'Fatalities due to virus and undertreatment', 'max'

y, y_label, y_method = 'virus_related_deaths', 'Fatalities due to virus and undertreatment', 'max'
# y, y_label, y_method = 'inactive_count', 'Maximum fraction of economically inactive population', 'max'
# y, y_label, y_method = 'inactive_count', 'Average fraction of economically inactive population', 'average'
# y, y_label, y_method = 'loss_in_output', 'Maximum loss in output', 'max'
# y, y_label, y_method = 'loss_in_output', 'Average loss in output', 'average'

monte_carlo_plot_cbar(x, x_label, x_method, y, y_label, y_method, color_by, color_by_label,
                      by_household_type=False, drop_after_time=None,
                      save=True, timestamp=True, csv_path=csv_path)

monte_carlo_plot_cbar(x, x_label, x_method, y, y_label, y_method, color_by, color_by_label,
                      by_household_type=True, drop_after_time=None,
                      save=True, timestamp=True, csv_path=csv_path)
