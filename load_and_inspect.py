import pandas as pd

# Load the CSV file
df = pd.read_csv("ssd-rainfall-adm2-full.csv")

# Print dataset shape
print("Shape of dataset:", df.shape)

# Print column names
print("\nColumn names:\n", df.columns)

# Print first few rows
print("\nSample data:\n", df.head())

# Check for missing values
print("\nMissing values:\n", df.isnull().sum())
