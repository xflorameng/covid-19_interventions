import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.tree import DecisionTreeRegressor as dtr
from dtreeviz.trees import dtreeviz


# Dataframe with ZIP code-level data from New York City
df = pd.read_csv('../data/combined_nyc_data.csv')
column_names = {
    'Percent Over 65': 'Age 65+',
    'Percent Non White': 'Non-white',
    'Covid Infection Rate': 'COVID-19 Infection',
    'Covid Death Rate': 'COVID-19 Deaths per 100,000 Population',
    'Eviction Rates': 'Eviction',
    'Unemployment Rate 2019': 'Unemployment 2019',
    'Percent Commuting': 'Commuting',
    'Percent Worked From Home': 'Remote work',
    'Health Insurance Coverage': 'Health insurance',
    'Percent Below Poverty': 'Below Poverty Line'
}
df.rename(columns=column_names, inplace=True)
df.dropna(inplace=True)
df['Increase in Unemployment Rate'] = df['Unemployment 2020 (Projected)'] - df['Unemployment 2019']
df['Income'] = df['Income'] / 1000
df['Overcrowding'] = df['Overcrowding'] * 100
df['Eviction'] = df['Eviction'] * 100
df['Non-white'] = df['Non-white'] * 100
df['Health insurance'] = df['Health insurance'] * 100

def trainDecisionTree(df, FeaturesToUse, FeatureLabels, predictionVariable, target_name, max_depth=6,
                      min_samples_leaf=9, plot=True, plot_tree=True, save=False, filename_tag=''):
    """
    - df is the dataframe with New York City data
    - FeaturesToUse is a list of variables we want to use to predict the predictionVariable
    - predicitonVariable is the variable we want to predict
    - max depth is the depth of the decision tree
    """

    assert predictionVariable not in FeaturesToUse, 'Remove the predictionVariable from FeaturesToUse'

    if len(FeaturesToUse) == 1:
        xTrain = df[FeaturesToUse[0]].to_numpy().reshape(-1, 1)
    else:
        xTrain = df[FeaturesToUse].to_numpy()

    yTrain = df[predictionVariable].to_numpy()

    regr = dtr(min_samples_leaf=min_samples_leaf, max_depth=max_depth)
    model = regr.fit(xTrain, yTrain)

    print(f'Feature importance: {np.round(model.feature_importances_, 2)}')

    if plot:
        plt.style.use('seaborn-white')
        plt.figure(figsize=(3.5, 4))
        fontsize = 14
        labelsize = 14
        plt.bar(FeaturesToUse, np.abs(model.feature_importances_), color=mcolors.CSS4_COLORS['cornflowerblue'])
        plt.xticks(range(len(FeaturesToUse)), FeaturesToUse, rotation=90, ha='center')
        plt.tick_params(axis='both', labelsize=labelsize)
        plt.ylabel('Feature importance', fontsize=fontsize)
        if save:
            plt.savefig(f'feature_importance{filename_tag}.pdf', bbox_inches='tight')
        plt.show()

    if plot_tree:
        plt.figure()
        viz = dtreeviz(regr,
                       xTrain,
                       yTrain,
                       target_name=target_name,
                       feature_names=FeatureLabels,
                       ticks_fontsize=10
                       )
        if save:
            viz.save(f'decision_tree{filename_tag}.svg')
        plt.show()

    return model

FeaturesToUse = ['Income', 'Age 65+', 'Non-white', 'Overcrowding',
                 'Remote work', 'Health insurance', 'Eviction', 'Commuting']
FeatureLabels = [r'Income ($10^3$ US\$)', 'Age 65+ (%)', 'Non-white (%)', 'Overcrowding (%)',
                 'Remote work (%)', 'Health insurance (%)', 'Eviction (%)', 'Commuting (%)']

predictionVariable = 'COVID-19 Deaths per 100,000 Population'
target_name = r'Deaths per $10^5$ people'  # This name will be displayed at the leaf nodes

# Full decision tree
dt_covid = trainDecisionTree(df, FeaturesToUse, FeatureLabels, predictionVariable, target_name,
                             save=True, filename_tag='_full')

df_worst = df.loc[(df['Income'] < 122.2) & (df['Age 65+'] >= 17.85)]  # Worst section of the tree
df_best = df.loc[(df['Income'] >= 122.2) & (df['Overcrowding'] >= 3.72)]  # Best section of the tree

print(f'[Worst segment] 2019 Unemployment rate {round(df_worst["Unemployment 2019"].mean(), 2)}')
print(f'[Best segment] 2019 Unemployment rate {round(df_best["Unemployment 2019"].mean(), 2)}')
print(f'[Worst segment] Increase in unemployment rate '
      f'{round(df_worst["Unemployment 2020 (Projected)"].mean() - df_worst["Unemployment 2019"].mean(), 2)}')
print(f'[Best segment] Increase in unemployment rate '
      f'{round(df_best["Unemployment 2020 (Projected)"].mean() - df_best["Unemployment 2019"].mean(), 2)}')
print(f'[Worst segment] Eviction rate {round(df_worst["Eviction"].mean(), 2)}')
print(f'[Best segment] Eviction rate {round(df_best["Eviction"].mean(), 2)}')

# Pruned decision tree
trainDecisionTree(df, FeaturesToUse, FeatureLabels, predictionVariable, target_name, max_depth=3, plot=False,
                  save=True, filename_tag='_pruned')
