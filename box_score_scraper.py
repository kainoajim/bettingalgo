import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# Load the game URLs from the CSV file (you can adjust the path)
df_game_urls = pd.read_csv('data/nba_game_urls.csv')
game_urls = df_game_urls['Game_URL'].tolist()

# List to store team stats data
team_stats_data = pd.DataFrame()

# Function to scrape both basic and advanced team stats for a single game
def scrape_team_stats(game_url):
    print(f"Scraping {game_url}...")
    match = game_url[-17:-5]
    print(match)
    
    # Send a request to the box score page
    response = requests.get(game_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # This section of code is to get the abbreviations for the teams
    # Find the meta tag that contains the og:description property
    meta_tag = soup.find('meta', {'property': 'og:description'})
    if meta_tag:
        # Extract the content of the meta tag
        content = meta_tag['content']       
        # Example: "PHI (117) vs BOS (126). Get the box score, shot charts and play by play summary..."
        # Split by ' vs ' to get the team abbreviations
        teams = content.split(' ')[0], content.split(' ')[3]
        team1 = teams[0]  # away team
        team2 = teams[1]  # home team

        print(f"Team 1: {team1}, Team 2: {team2}")
    else:
        print("Meta tag with og:description not found.")
        return
    
    # the selectors are the names of the tables
    selectors = [f'box-{team1}-game-basic', f'box-{team2}-game-basic', f'box-{team1}-game-advanced', f'box-{team2}-game-advanced']

    dfs = []
    for selector in selectors:
            table = soup.find('table', { 'id': selector })
            if table:
                raw_df = pd.read_html(str(table))[0]
                df = _process_box(raw_df)
                team_totals_row = df[df['PLAYER'] == 'Team Totals']
                # removes first column
                team_totals_row = team_totals_row.drop(team_totals_row.columns[0], axis=1)
                # add to team totals
                if not team_totals_row.empty:
                    dfs.append(team_totals_row)
                else:
                    print(f"Team Totals row not found for {selector}")
            else:
                print(f"Table {selector} not found")

    # print(dfs)
    team1_basic = dfs[0]
    team2_basic = dfs[1]
    team1_advanced = dfs[2]
    team2_advanced = dfs[3]

    # Add a winner variable
    team1_points = float(team1_basic['PTS'].values[0])  # Get the points for Team 1
    team2_points = float(team2_basic['PTS'].values[0])  # Get the points for Team 2

    # Determine the winner
    if team1_points > team2_points:
        team1_basic['WIN'] = 1
        team2_basic['WIN'] = 0
    elif team2_points > team1_points:
        team1_basic['WIN'] = 0
        team2_basic['WIN'] = 1
    else:  # In case of a tie (if needed)
        team1_basic['WIN'] = 0
        team2_basic['WIN'] = 0

    # Combine the basic and advanced stats for each team by concatenating the columns
    team1_combined = pd.concat([team1_basic.reset_index(drop=True), team1_advanced.reset_index(drop=True)], axis=1)
    team2_combined = pd.concat([team2_basic.reset_index(drop=True), team2_advanced.reset_index(drop=True)], axis=1)

    # Clean up NaN values by replacing them with 0 (or you can use any other value like None)
    team1_combined = team1_combined.fillna(0)
    team2_combined = team2_combined.fillna(0)
    #insert game ID into stats
    team1_combined.insert(0, 'gameID', match)
    team2_combined.insert(0, 'gameID', match)
    # insert team name into stats
    team1_combined.insert(0, 'Team', team1)
    team2_combined.insert(0, 'Team', team2)

    # Ensure data is 2D before appending
    print(f"Team 1 combined shape: {team1_combined.shape}")
    print(f"Team 2 combined shape: {team2_combined.shape}")

    global team_stats_data
    team_stats_data = pd.concat([team_stats_data, team1_combined, team2_combined], ignore_index=True)
    
def _process_box(df):
    """ Perform basic processing on a box score - common to both methods

    Args:
        df (DataFrame): the raw box score df

    Returns:
        DataFrame: processed box score
    """
    df.columns = list(map(lambda x: x[1], list(df.columns)))
    df.rename(columns = {'Starters': 'PLAYER'}, inplace=True)
    if 'Tm' in df:
        df.rename(columns = {'Tm': 'TEAM'}, inplace=True)
    reserve_index = df[df['PLAYER']=='Reserves'].index[0]
    df = df.drop(reserve_index).reset_index().drop('index', axis=1) 
    # remove first column
    # df = df.drop(df.columns[0], axis=1)
    # Add the team name as the first column
    # df.insert(0, 'Team', team_name)
    return df


# Scrape the stats for all games
for i, game_url in enumerate(game_urls[:3]):
    scrape_team_stats(game_url)
    
    # Add a small delay to avoid overwhelming the server
    time.sleep(2)

# Define column headers for basic and advanced stats (you can adjust these based on your needs)
columns = [
    'Team', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS',
    'MP', 'TS%', 'eFG%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'ORtg', 'DRtg', 'BPM'
]

# Convert the team stats data into a DataFrame
# print(f"Team 1 combined shape: {team_stats_data.shape}")
# df_team_stats = pd.DataFrame(team_stats_data, columns=columns)

# Save the combined basic and advanced stats to a CSV file
team_stats_data.to_csv('data/nba_team_stats.csv', index=False)

print(f"Scraped and saved team stats for {len(game_urls)} games.")
