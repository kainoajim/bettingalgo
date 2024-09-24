import pandas as pd

# Load the CSV containing team stats (both basic and advanced)
df = pd.read_csv('data/nba_team_stats.csv')

# Display the first few rows to inspect the data
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Fill or drop missing values (you can fill with mean or median, or remove rows if necessary)
df = df.dropna()  # For simplicity, drop rows with missing values

# Convert any necessary columns to numeric (e.g., points, advanced stats)
numeric_columns = ['PTS', 'TS%', 'ORB%', 'DRB%', 'ORtg', 'DRtg', 'MP']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Remove rows with invalid data (optional)
df = df.dropna()

# Split the data into features (X) and targets (y)
X = df[['TS%', 'ORB%', 'DRB%', 'ORtg', 'DRtg']]  # Example: Use relevant stats as features
y = df['PTS']  # Target variable: Points scored by the team
