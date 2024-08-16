import pandas as pd

# Load the dataset
file_path = 'Amazon Customer Behavior Survey.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("Initial Data Preview:")
print(data.head())

# Handling missing values
# Option 1: Drop rows with any missing values
data_cleaned = data.dropna()

# Option 2: Fill missing values with a specific value, e.g., 0 or 'Unknown'
# data_cleaned = data.fillna(0)  # or data.fillna('Unknown')

# Remove duplicates
data_cleaned = data_cleaned.drop_duplicates()

# Convert data types if necessary (example: converting a column to datetime)
# data_cleaned['date_column'] = pd.to_datetime(data_cleaned['date_column'])

# Convert categorical columns to category type
# data_cleaned['category_column'] = data_cleaned['category_column'].astype('category')

# Display the cleaned data
print("Cleaned Data Preview:")
print(data_cleaned.head())

# Save the cleaned dataset to a new CSV file
cleaned_file_path = 'Cleaned_Amazon_Customer_Behavior_Survey.csv'
data_cleaned.to_csv(cleaned_file_path, index=False)

print(f"Cleaned dataset saved to {cleaned_file_path}")
