from src.models import monte_carlo_contourf


csv_path = '../results/stats/transmission_probs_reopen/'

x, x_label = 'household_transmission_prob', 'Household transmission probability'

y, y_label = 'other_transmission_prob', 'Other transmission probability'

color_by, color_by_label, color_by_method = 'total_deaths', 'Fatalities', 'max'
# color_by, color_by_label, color_by_method = 'deaths_of_despair', 'Deaths of Despair', 'max'
# color_by, color_by_label, color_by_method = ('virus_related_deaths', 'Fatalities due to Virus\nand Undertreatment',
#                                              'max')
# color_by, color_by_label, color_by_method = ('inactive_count', 'Maximum Fraction of\nEconomically Inactive Population',
#                                              'max')
# color_by, color_by_label, color_by_method = ('inactive_count', 'Average Fraction of\nEconomically Inactive Population',
#                                              'average')
# color_by, color_by_label, color_by_method = 'loss_in_output', 'Maximum Loss in Output', 'max'
# color_by, color_by_label, color_by_method = 'loss_in_output', 'Average Loss in Output', 'average'

monte_carlo_contourf(x, x_label, y, y_label, color_by, color_by_label, color_by_method,
                     drop_after_time=None, figsize=(4, 4), save=True, timestamp=True, csv_path=csv_path)
