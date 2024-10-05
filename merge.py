import os
import pandas as pd

# 1. Load the 20 Newsgroups human-written text files into human_df

# Directory where your 20 Newsgroups .txt files are stored
newsgroups_dir = '/Users/aether/Downloads/DATA/archive/'  # Update with your actual path

# Initialize lists to hold the texts and labels
human_texts = []
labels = []

# Loop through each .txt file in the directory and read its contents
for filename in os.listdir(newsgroups_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(newsgroups_dir, filename)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            text = file.read()
            human_texts.append(text)
            labels.append(0)  # Label 0 for human-written text

# Create a DataFrame for human text
human_df = pd.DataFrame({'text': human_texts, 'label': labels})

# 2. Load the AI-generated text dataset from the Parquet file into ai_generated_df
ai_generated_df = pd.read_parquet('combined_dataset.parquet')

# Add a label column to the AI-generated DataFrame, setting all values to 1 (AI-generated text)
ai_generated_df['label'] = 1

# Ensure that the AI-generated dataset has only the 'text' and 'label' columns
ai_generated_df = ai_generated_df[['text', 'label']]

# 3. Merge the human-written and AI-generated text DataFrames
# Both DataFrames must have the same structure: 'text' and 'label'
combined_df = pd.concat([human_df, ai_generated_df], ignore_index=True)

# Display the first few rows of the combined dataset
print(combined_df.head())

# 4. Save the combined dataset to a CSV file with correct column names
combined_df.to_csv('final_combined_dataset.csv', index=False)
# Or save as Parquet if you prefer
# combined_df.to_parquet('final_combined_dataset.parquet', index=False)
