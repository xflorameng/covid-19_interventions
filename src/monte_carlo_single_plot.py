from src.models import monte_carlo_single_plot


csv_path = '../results/stats/worker_staying_at_home_prob/'

x, x_label, x_method = 'worker_staying_at_home_prob', 'Lockdown level', None

y, y_label, y_method = 'total_deaths', 'Fatalities', 'max'
# y, y_label, y_method = 'deaths_of_despair', 'Deaths of Despair', 'max'
# y, y_label, y_method = 'virus_related_deaths', 'Fatalities due to virus and undertreatment', 'max'
# y, y_label, y_method = 'inactive_count', 'Maximum fraction of economically inactive population', 'max'
# y, y_label, y_method = 'inactive_count', 'Average fraction of economically inactive population', 'average'
# y, y_label, y_method = 'loss_in_output', 'Maximum loss in output', 'max'
# y, y_label, y_method = 'loss_in_output', 'Average loss in output', 'average'

monte_carlo_single_plot(x, x_label, x_method, y, y_label, y_method,
                        std=True, drop_after_time=None, save=True, timestamp=True, csv_path=csv_path)
