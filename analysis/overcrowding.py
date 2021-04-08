import warnings

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns


# COVID-19 deaths data from Hopkins Population Center (HPC)
covid_deaths_date = '20200731'

covid = pd.read_csv('hpc_data_hub/Pandemic/casesAndDeaths.csv',
                    index_col=['fips'], usecols=['fips', 'stname', 'ctyname', 'deaths_' + covid_deaths_date])

covid.rename_axis('county_id')
column_names = {
    'stname': 'state',
    'ctyname': 'county_name',
    'deaths_' + covid_deaths_date: 'covid_deaths'
}
covid.rename(columns=column_names, inplace=True)
covid.dropna(inplace=True)
if not covid.index.is_unique:
    warnings.warn('Duplicate FIPS code detected')
covid['county_name'] = covid['county_name'].map(lambda x: x.replace(' County', '', 1))

# Population data from Hopkins Population Center (HPC)
population = pd.read_csv('hpc_data_hub/Pandemic/mobility.csv',
                         index_col=['fips'], usecols=['fips', 'stname', 'ctyname', 'county_population_2018'])

population.rename_axis('county_id')
column_names = {
    'stname': 'state',
    'ctyname': 'county_name',
    'county_population_2018': 'county_population'
}
population.rename(columns=column_names, inplace=True)
population.dropna(inplace=True)
if population['county_population'].le(0).sum() > 0:
    warnings.warn('Nonpositive population detected')
if not population.index.is_unique:
    warnings.warn('Duplicate FIPS code detected')
population['county_name'] = population['county_name'].map(lambda x: x.replace(' County', '', 1))

# Household overcrowding data
overcrowding = pd.read_csv('overcrowded_households_combined.csv', index_col=['county_id'])

# Rurality data
rurality = pd.read_csv('urban_rural_classification_combined.csv', index_col=['county_id'])

# Combining data
df = pd.concat([covid, population, overcrowding, rurality], axis=1, join='inner')
df.dropna(inplace=True)
county_name = df['county_name']
county_name_mask = county_name.eq(county_name.iloc[:, 0], axis=0).all(axis=1)
df = df.loc[county_name_mask, :]
state = df['state']
state_mask = state.eq(state.iloc[:, 0], axis=0).all(axis=1)
df = df.loc[state_mask, :]
df = df.loc[:, ~df.columns.duplicated()]
numeric_columns = list(df.columns)
numeric_columns.remove('county_name')
numeric_columns.remove('state')
numeric_columns.remove('county_rurality')
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
df.dropna(inplace=True)
df['covid_deaths_normalized'] = df['covid_deaths'] / df['county_population'] * 1e5
df['percent_household_overcrowding'] = df['percent_household_overcrowding'] * 100

# Filtering counties by rurality
df = df.loc[df['percent_rural_population'] <= 5]
county_rurality = 'mostly_urban'
# df = df.loc[(df['percent_rural_population'] >= 45) & (df['percent_rural_population'] <= 55)]
# county_rurality = 'urban_rural'
# df = df.loc[df['percent_rural_population'] == 100]
# county_rurality = 'rural'

# Filtering states by the number of counties
n_largest_states = 4
n_counties_by_state = df.groupby('state').size()
df = df.loc[df['state'].isin(n_counties_by_state.nlargest(n_largest_states).index)]

n_cols = n_largest_states
fontsize = 18
labelsize = fontsize
plt.style.use('seaborn-white')
g = sns.lmplot(x='percent_household_overcrowding', y='covid_deaths_normalized', data=df,
               col='state', col_wrap=n_cols, robust=True, ci=95, n_boot=5000, height=4.5,
               scatter_kws={'s': 160, 'color': mcolors.CSS4_COLORS['lightcoral']},
               line_kws={'color': mcolors.CSS4_COLORS['darkred'], 'linewidth': 5})
plt.xlim(0, 13)
plt.ylim(-20, 400)
axes = g.axes.flatten()
for i in range(len(axes)):
    axes[i].set_title(df.state.unique()[i], fontsize=fontsize)
    axes[i].set_xlabel('Percent of household overcrowding', fontsize=fontsize)
    axes[i].tick_params(labelsize=labelsize)
    if i % n_cols == 0:
        axes[i].set_ylabel(r'Deaths per $10^5$ people', fontsize=fontsize)
plt.savefig(f'household_overcrowding_covid_deaths_by_state_{county_rurality}.pdf',
            transparent=True, bbox_inches='tight')
