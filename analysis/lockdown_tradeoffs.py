import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.colors as mcolors
from sklearn.linear_model import LinearRegression as lr


def fitExponential(xvals, yvals):
    ylog = np.log(yvals)
    l = lr()
    l.fit(xvals.reshape(-1, 1), ylog)
    k = np.exp(l.intercept_)
    alpha = l.coef_
    return k, alpha

# County-level data
df = pd.read_csv('../data/combined_county_data.csv')
df.dropna(inplace=True)
dfpoor = df.loc[df['Income'] < 70000]
dfrich = df.loc[df['Income'] > 80000]
poor_mobility = dfpoor['residential_percent_change_from_baseline'].to_numpy()
rich_mobility = dfrich['residential_percent_change_from_baseline'].to_numpy()
poor_unemployment_deaths = dfpoor['Projected Deaths From Unemployment'].to_numpy()
rich_unemployment_deaths = dfrich['Projected Deaths From Unemployment'].to_numpy()
poor_covid_deaths = dfpoor['Covid Death Rate (Delayed After Lockdown)'].to_numpy()
rich_covid_deaths = dfrich['Covid Death Rate (Delayed After Lockdown)'].to_numpy()

# Linear regression of deaths from unemployment on the mobility change
lpoor = lr().fit(poor_mobility.reshape(-1, 1), poor_unemployment_deaths)
xvalspoor = np.linspace(10, 30, 100)
yvals_poor_unemployment = xvalspoor * lpoor.coef_ + lpoor.intercept_
lrich = lr().fit(rich_mobility.reshape(-1, 1), rich_unemployment_deaths)
xvalsrich = np.linspace(10, 30, 100)
yvals_rich_unemployment = xvalsrich * lrich.coef_ + lrich.intercept_

# Linear regression of logarithm of COVID-19 deaths on the mobility change
k_poor, alpha_poor = fitExponential(poor_mobility, poor_covid_deaths)
yvals_poor_covid = k_poor * np.exp(alpha_poor * xvalspoor)
k_rich, alpha_rich = fitExponential(rich_mobility, rich_covid_deaths)
yvals_rich_covid = k_rich * np.exp(alpha_rich * xvalsrich)

# Plot results
rcParams.update({'figure.autolayout': True})
marker_size = 120
linewidth = 4
fontsize = 16
labelsize = 16
figsize = (6, 4)
blue = mcolors.CSS4_COLORS['cornflowerblue']
red = mcolors.CSS4_COLORS['lightcoral']

plt.figure(figsize=figsize)
plt.scatter(rich_mobility, rich_covid_deaths, label='COVID-19', s=marker_size, color=red, alpha=.7)
plt.scatter(rich_mobility, rich_unemployment_deaths, label='Unemployment', s=marker_size, color=blue, alpha=.7)
plt.plot(xvalsrich, yvals_rich_covid, linewidth=linewidth, color=mcolors.CSS4_COLORS['brown'])
plt.plot(xvalsrich, yvals_rich_unemployment, linewidth=linewidth, color=mcolors.CSS4_COLORS['royalblue'])
plt.xlim(10, 30)
plt.ylim(0, 31)
# plt.legend(prop={'size': fontsize}, loc='upper right')
plt.ylabel(r'Deaths per $10^5$ people', fontsize=fontsize)
plt.title('Death by cause: Rich counties', fontsize=fontsize)
plt.xlabel('Percent change of people staying at home', fontsize=fontsize)
plt.tick_params(axis='both', labelsize=labelsize)
plt.savefig('outputs/mobility_death_by_cause_rich.pdf', transparent=True, bbox_inches='tight')
plt.show()

plt.figure(figsize=figsize)
plt.scatter(poor_mobility, poor_covid_deaths, label='COVID-19', s=marker_size, color=red, alpha=.7)
plt.scatter(poor_mobility, poor_unemployment_deaths, label='Unemployment', s=marker_size, color=blue, alpha=.7)
plt.plot(xvalspoor, yvals_poor_covid, linewidth=linewidth, color=mcolors.CSS4_COLORS['brown'])
plt.plot(xvalspoor, yvals_poor_unemployment, linewidth=linewidth, color=mcolors.CSS4_COLORS['royalblue'])
plt.xlim(10, 30)
plt.ylim(0, 31)
plt.legend(prop={'size': fontsize}, loc='upper right')
plt.ylabel(r'Deaths per $10^5$ people', fontsize=fontsize)
plt.title('Death by cause: Poor counties', fontsize=fontsize)
plt.xlabel('Percent change of people staying at home', fontsize=fontsize)
plt.tick_params(axis='both', labelsize=labelsize)
plt.savefig('outputs/mobility_death_by_cause_poor.pdf', transparent=True, bbox_inches='tight')
plt.show()

plt.figure(figsize=figsize)
plt.title('Combined deaths: Poor counties', fontsize=fontsize)
plt.plot(xvalspoor, yvals_poor_covid + yvals_poor_unemployment, linewidth=linewidth,
         color=mcolors.CSS4_COLORS['darkorange'])
plt.xlim(10, 30)
plt.xlabel('Percent change of people staying at home', fontsize=fontsize)
plt.ylabel(r'Deaths per $10^5$ people', fontsize=fontsize)
plt.tick_params(axis='both', labelsize=labelsize)
plt.savefig('outputs/mobility_combined_deaths.pdf', transparent=True, bbox_inches='tight')
plt.show()
