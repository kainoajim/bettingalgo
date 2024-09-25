import pandas as pd

# Load your dataset
df = pd.read_csv('data/nba_team_stats.csv')

def preprocess_data_for_training(df):
    # Sort the data by team and gameID
    df = df.sort_values(by=['Team', 'gameID'])
    
    # List of stats for which we want to calculate rolling averages
    rolling_columns = ['FG%', '3P%', 'FT%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'TS%', 'FTr', 'ORtg', 'DRtg']
    
    # Create an empty DataFrame to store the processed data
    processed_df = pd.DataFrame()

    # Group by team to calculate rolling averages separately for each team
    for team, team_df in df.groupby('Team'):
        team_df = team_df.copy()  # Avoid modifying the original data

        # Calculate rolling averages over the last 10 games for each key stat
        for col in rolling_columns:
            team_df[f'Rolling_{col}'] = team_df[col].rolling(window=10, min_periods=10).mean()

        # Calculate rolling wins over the last 10 games (if you have a 'Win' column)
        team_df['Rolling_Wins'] = team_df['WIN'].rolling(window=10, min_periods=10).sum()

        # Shift the rolling averages by 1 to use them as inputs for the next game's prediction
        rolling_feature_cols = [f'Rolling_{col}' for col in rolling_columns] + ['Rolling_Wins']
        team_df[rolling_feature_cols] = team_df[rolling_feature_cols].shift(1)

        # Filter out the first 10 games since they don’t have enough history for rolling stats
        team_df = team_df.dropna(subset=rolling_feature_cols)

        # Append the processed team data to the main DataFrame
        processed_df = pd.concat([processed_df, team_df], ignore_index=True)

    # Drop unnecessary columns such as 'Game_Date' and any other non-predictive data
    # processed_df = processed_df.drop(columns=['gameID', 'Team', 'WIN'])  # You can modify based on your needs

    return processed_df

# Run the preprocessing function
processed_df = preprocess_data_for_training(df)

# Save the preprocessed data to a new CSV
processed_df.to_csv('data/nba_preprocessed_for_training.csv', index=False)

print("Data preprocessing for training completed and saved to 'data/nba_preprocessed_for_training.csv'.")
