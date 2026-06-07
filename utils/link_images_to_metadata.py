import pandas as pd

# Load the metadata CSVs (from downloaded Kaggle dataset)
artwork_df = pd.read_csv('data/historic-art/complete/artwork_dataset.csv')
info_df = pd.read_csv('data/historic-art/complete/info_dataset.csv')

# Image directory inside the downloaded dataset
image_dir = 'data/historic-art/complete/artwork'  # 45,574 JPG images named by ID

# Display column names and preview the data
print("Artwork columns:", artwork_df.columns.tolist())
print("\nFirst few rows of artwork_df:")
print(artwork_df.head())


import os

# Create a new column for the image path using the ID
artwork_df['image_path'] = artwork_df['ID'].apply(lambda x: os.path.join(image_dir, f"{x}.jpg"))

# Show a sample to verify
print("\nSample image paths:")
print(artwork_df[['ID', 'image_path']].head())


print("artwork_df columns:", artwork_df.columns.tolist())
print("info_df columns:", info_df.columns.tolist())


# Merge artwork and artist info on the 'artist' column
merged_df = pd.merge(artwork_df, info_df, how='left', on='artist')

# Preview the merged data
print("Merged DataFrame columns:", merged_df.columns.tolist())
print("\nSample merged rows:")
print(merged_df.head())


# Only keep rows where the image exists and artist label is present
import os

# Check if image file exists
merged_df['image_exists'] = merged_df['image_path'].apply(os.path.exists)
clean_df = merged_df[merged_df['image_exists'] & merged_df['artist'].notnull()].copy()

print(f"Total artworks after cleaning: {len(clean_df)}")
print(clean_df[['ID', 'artist', 'image_path']].head())


print(clean_df['artist'].value_counts().head(20))  # Show top 20 artists


# Save the merged DataFrame to a CSV file for inspection/sharing
merged_df.to_csv('data/merged_artwork_metadata.csv', index=False)
artwork_df.to_csv('data/mapped.csv', index=False)  # also save as mapped.csv for EDA scripts

print("Merged DataFrame saved as 'data/merged_artwork_metadata.csv'")
print("Also saved as 'data/mapped.csv' for EDA and training scripts")
