import pandas as pd

# Define your lists of column names
numerical_columns = [
    'Mean_X', 'Std Dev_X', 'Energy_X', 'Entropy_X', 'Peaks_X',
    'Mean_Y', 'Std Dev_Y', 'Energy_Y', 'Entropy_Y', 'Peaks_Y',
    'Mean_Z', 'Std Dev_Z', 'Energy_Z', 'Entropy_Z', 'Peaks_Z',
    'Mean_Mixed', 'Std Dev_Mixed', 'Energy_Mixed', 'Entropy_Mixed', 'Peaks_Mixed'
]
categorical_columns = ['category']

# Sample data for testing (replace this with your actual DataFrame)
df = pd.read_csv('artifacts/train.csv')

# Combine the lists of expected columns
expected_columns = numerical_columns + categorical_columns

# Check for missing columns
missing_columns = [col for col in expected_columns if col not in df.columns]

if not missing_columns:
    print("All expected columns are present.")
else:
    print("The following columns are missing:", missing_columns)
