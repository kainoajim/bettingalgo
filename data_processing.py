import pandas as pd

# Load the full dataset
df = pd.read_csv('data/nba_team_stats_2022-2024.csv')

betting_data = pd.read_csv('data/nba_spreads_totals.csv')

result_df = df[['Team', 'gameID', 'HOME', 'PTS']]

# List of stats for which we want to calculate rolling averages
rolling_columns = [
    'FG', 'FGA', 'FG%', '3P', '3PA', '3P%',
    'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 
    'TS%', 'eFG%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%', 'AST%', 
    'STL%', 'BLK%', 'TOV%', 'USG%', 'ORtg', 'DRtg'
]


# Function to calculate rolling averages for each team
def calculate_rolling_averages(df):
    # Sort the data by team and game date
    df = df.sort_values(by=['Team', 'gameID'])
    
    # Create an empty DataFrame to store the rolling averages
    rolling_df = pd.DataFrame()
    
    # Group by team to calculate rolling averages for each team separately
    for team, team_df in df.groupby('Team'):
        team_df = team_df.copy()
        
        # Calculate rolling averages for each stat
        for col in rolling_columns:
            team_df[f'Rolling_{col}'] = team_df[col].rolling(window=10, min_periods=10).mean()
        
        # Calculate rolling wins (as a recent form indicator)
        team_df['Rolling_Wins'] = team_df['WIN'].rolling(window=10, min_periods=10).sum()
        
        # Shift the rolling averages to ensure the stats are for the previous 10 games
        rolling_feature_cols = [f'Rolling_{col}' for col in rolling_columns] + ['Rolling_Wins']
        team_df[rolling_feature_cols] = team_df[rolling_feature_cols].shift(1)
        
        # Drop the first 10 games, as they don't have enough history for rolling averages
        team_df = team_df.dropna(subset=rolling_feature_cols)

        # Keep only the rolling columns and necessary identifiers
        team_df = team_df[['Team', 'gameID', 'HOME'] + rolling_feature_cols]
        
        # Append the rolling averages for this team to the main DataFrame
        rolling_df = pd.concat([rolling_df, team_df], ignore_index=True)
    
    return rolling_df

# Calculate rolling averages for each team
rolling_df = calculate_rolling_averages(df)

# Merge home and away teams into a single row per game
final_dataset = pd.DataFrame()

# Assume that the home and away teams are known through GameID
for game_id in rolling_df['gameID'].unique():
    game_data = rolling_df[rolling_df['gameID'] == game_id]
    
    # Separate home and away teams
    home_team = game_data[game_data['HOME'] == 1]
    away_team = game_data[game_data['HOME'] == 0]

    # home_team_name = home_team['Team']
    # away_team_name = away_team['Team']
    # print(home_team_name)

    game_result = result_df[result_df['gameID'] == game_id]
    home_result = game_result[game_result['HOME'] == 1]
    away_result = game_result[game_result['HOME'] == 0]

    # Add prefixes to column names for home and away teams
    home_team.columns = [f'Home_{col}' if col not in ['gameID', 'Team'] else col for col in home_team.columns]
    away_team.columns = [f'Away_{col}' if col not in ['gameID', 'Team'] else col for col in away_team.columns]
    
    # Combine their rolling averages and target columns into one row
    combined_row = pd.concat([home_team.reset_index(drop=True), away_team.reset_index(drop=True)], axis=1)

    # Add the actual scores as target variables
    # combined_row['Home_Team'] = home_team_name
    # combined_row['Away_Team'] = away_team_name

    # Add the actual scores as target variables
    combined_row['Home_Score'] = home_result['PTS'].values[0]
    combined_row['Away_Score'] = away_result['PTS'].values[0]


    
    # Append to the final dataset
    final_dataset = pd.concat([final_dataset, combined_row], ignore_index=True)

final_dataset = final_dataset.round(6)

# comment this out if you want to bring back the team and gameID for checking
final_dataset = final_dataset.drop(columns=['Team'])

# Remove duplicate 'gameID' column from final_dataset
final_dataset = final_dataset.loc[:, ~final_dataset.columns.duplicated()]

# print(final_dataset.columns[final_dataset.columns.duplicated()])

# print(betting_data.columns[betting_data.columns.duplicated()])


# Merge the final_dataset with the betting data using 'gameID' as the key
merged_dataset = pd.merge(final_dataset, betting_data[['gameID', 'spread', 'total']], on='gameID', how='inner')

# merged_dataset = merged_dataset.drop(columns=['gameID'])

# Now the dataset is ready for model training
merged_dataset.to_csv('data/nba_model_training_data.csv', index=False)
