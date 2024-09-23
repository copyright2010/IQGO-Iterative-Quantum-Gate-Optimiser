import matplotlib.pyplot as plt
import pandas as pd
import openml
from scipy.stats import chi2_contingency
import seaborn as sns
import numpy as np
from statsmodels.tsa.stattools import adfuller

def adf_test(timeseries):
    print('Results of Augmented Dickey-Fuller Test:')
    result = adfuller(timeseries, autolag='AIC')  # AIC is used to select the best lag length
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')


valid_dids_classification = ([13, 59, 4, 15, 40710, 43, 1498])
# 15
dataset = openml.datasets.get_dataset(756)
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="array", target=dataset.default_target_attribute
)
# 724, 774, 43
print(dataset)
print(X)
df = X
arr = pd.DataFrame(df)
from collections import Counter

print(Counter(y))

for column in arr.columns:
    print(f'\nADF Test for {column}:')
    adf_test(arr[column])

# for column in arr.columns:
#     sns.boxplot(arr[column])
#     plt.show()

varr, std, mean, skew, kurtosis, median, mode = [], [], [] ,[] ,[] ,[] ,[] 

for column in arr.columns:
    varr.append(np.var(arr[column]))
    std.append(np.std(arr[column]))
    mean.append(np.mean(arr[column]))
    skew.append((arr[column]).skew())
    kurtosis.append((arr[column]).kurtosis())
    median.append(np.median(arr[column]))
    mode.append((arr[column]).mode())

print('var',varr)
print('std',std)
print('mean',mean)
print('skew',skew)
print('kurtosis',kurtosis)
print('median',median)
print('mode',mode)

# plt.plot(X)
# plt.show()
# exit()

# # Feature Distribution Analysis for Numerical Features
# for column in df.columns:
#     plt.figure(figsize=(10, 4))
#     sns.histplot(df[column], kde=True)
#     plt.title(f'Distribution of {column}')
#     plt.show()

# #df = df.reset_index()
# for column in df.columns[0:3]:
#     print(column)
#     arr[column] = (df[column])

# print(arr['Age_of_patient_at_time_of_operation'])
# print(arr['Patients_year_of_operation'])
# print(arr[[0,1,2]].values)
# print(arr.corr())

# # Correlation Analysis for Numerical Features
# correlation_matrix = arr.corr()#df[['Age_of_patient_at_time_of_operation','Number_of_positive_axillary_nodes_detected']].corr()
# print(correlation_matrix)

# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", center=0)
# plt.title('Correlation Matrix for Numerical Features')
# plt.show()

# contingency_table = pd.crosstab(df, y)
# chi2, p, dof, expected = chi2_contingency(contingency_table)
# print(f"Chi-square test result for CategoricalFeature vs Target: p-value = {p}")
