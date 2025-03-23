import pandas as pd

# Load the dataset
dataset_path = 'biceps_curl_dataset.csv'
data = pd.read_csv(dataset_path)

# Display first few rows of the dataset
print(data.head())
