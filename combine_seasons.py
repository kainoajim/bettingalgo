import pandas as pd

# Load the two CSV files
file_1 = 'data/nba_team_stats_2022_2023.csv'  # Replace with your actual filename
file_2 = 'data/nba_team_stats_2024.csv'  # Replace with your actual filename

# Read the CSV files into DataFrames
df1 = pd.read_csv(file_1)
df2 = pd.read_csv(file_2)

# Concatenate the two DataFrames
combined_df = pd.concat([df1, df2], ignore_index=True)

# Remove rows with missing values (NaN) in any column
combined_df_cleaned = combined_df.dropna()

# Save the cleaned combined DataFrame to a new CSV file
combined_df_cleaned.to_csv('data/nba_team_stats_2022-2024.csv', index=False)

print(f"Combined data saved to 'data/combined_seasons_cleaned.csv'.")
