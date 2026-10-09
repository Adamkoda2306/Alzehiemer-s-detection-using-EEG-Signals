import pandas as pd

dataset = pd.read_csv('adhdata.csv')

print("Dataset Overview:")
print(dataset.head())

print("\nMissing Values:")
print(dataset.isnull().sum())

print("\nUnique Classes:")
print(dataset['Class'].unique())

print("\nNumber of Unique Classes:")
print(dataset['Class'].nunique())

print("\nNumber of Control samples:")
print((dataset['Class'] == 'Control').sum())

print("\nNumber of unique Control patients:")
print(dataset.loc[dataset['Class'] == 'Control', 'ID'].nunique())