import numpy as np
import pandas as pd

'''
You might think there will be three input units, but we actually need to transform the data first.
 rank feature is categorical, the numbers don't encode any sort of relative values.
 Rank 2 is not twice as much as rank 1, rank 3 is not 1.5 more than rank 2.
 Instead, we need to use dummy variables to encode rank, splitting the data into four new columns encoded with
 ones or zeros. Rows with rank 1 have one in the rank 1 dummy column, and zeros in all other columns.
 Rows with rank 2 have one in the rank 2 dummy column, and zeros in all other columns.
 And so on.
'''

admissions = pd.read_csv('binary.csv')

# Make dummy variables for rank
data = pd.concat([admissions, pd.get_dummies(admissions['rank'], prefix='rank')], axis=1)
data = data.drop('rank', axis=1)

# Standarize features
for field in ['gre', 'gpa']:
    mean, std = data[field].mean(), data[field].std()
    data.loc[:, field] = (data[field] - mean) / std

# Split off random 10% of the data for testing
np.random.seed(42)
sample = np.random.choice(data.index, size=int(len(data) * 0.9), replace=False)
data, test_data = data.ix[sample], data.drop(sample)

# Split into features and targets
features, targets = data.drop('admit', axis=1), data['admit']
features_test, targets_test = test_data.drop('admit', axis=1), test_data['admit']