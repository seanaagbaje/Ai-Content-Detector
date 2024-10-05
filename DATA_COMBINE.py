import pandas as pd
import os

# Directory where the shards are stored
shards_dir = '/Users/aether/Downloads/DATA/'  # Update with your actual path

# List all Parquet files in the directory
parquet_files = sorted([f for f in os.listdir(shards_dir) if f.endswith('.parquet')])

# Initialize an empty list to hold DataFrames
dataframes = []

# Load each shard and append to the list
for parquet_file in parquet_files:
    shard_path = os.path.join(shards_dir, parquet_file)
    df = pd.read_parquet(shard_path)
    dataframes.append(df)

# Concatenate all the DataFrames into a single DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# Display the combined DataFrame
print(combined_df.head())

# Optionally save the combined DataFrame to a single Parquet file or CSV
combined_df.to_parquet('combined_dataset.parquet')  # Save as a single Parquet file
# combined_df.to_csv('combined_dataset.csv', index=False)  # Save as a CSV file
