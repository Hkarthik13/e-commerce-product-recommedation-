import pandas as pd
import numpy as np

# Load dataset
data = pd.read_csv('Amazon Customer Behavior Survey.csv')

# Explore the dataset
print(data.head())
print(data.info())
print(data.describe())

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Assuming 'rating' is the interaction metric in the dataset
data['interaction'] = data['Rating_Accuracy']

# Verify the interaction column
print(data.head())
