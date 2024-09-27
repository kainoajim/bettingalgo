import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

# Function to convert team name to the abbreviation used in gameID
def get_team_abbreviation(team_name):
    """Convert full team name to the 3-letter abbreviation for gameID"""
    team_abbreviations = {
        'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN', 'Charlotte Hornets': 'CHA',
        'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE', 'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN',
        'Detroit Pistons': 'DET', 'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
        'LA Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM', 'Miami Heat': 'MIA',
        'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN', 'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK',
        'Oklahoma City Thunder': 'OKC', 'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
        'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS', 'Toronto Raptors': 'TOR',
        'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
    }
    return team_abbreviations.get(team_name, 'UNKNOWN')

def generate_game_id(game_date, home_team):
    """Generate the gameID using the format yyyymmdd0{hometeam abbreviation}"""
    date_str = game_date.strftime('%Y%m%d')
    home_abbreviation = get_team_abbreviation(home_team)
    return f"{date_str}0{home_abbreviation}"

# Function to scrape DraftKings home team spread and other relevant data
def scrape_game_info(url, game_date):
    """Scrapes home team name and spread from the page"""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Locate all game descriptions by targeting the div with the game description class
    games = soup.find_all('div', class_='OddsTableMobile_gameDescription__chcQf')

    data = []

    # Debug: print the number of games found
    print(f"Found {len(games)} game descriptions on the page.")

    # Loop through each game description to extract the home team and spread
    for game in games:
        try:
            # Debug: print the structure of the game div
            print("Game div structure:")
            print(game.prettify())  # This will output the structure of each game description div

            # Extract the teams from the <div> tag with class "h5" (e.g., "Away Team vs Home Team")
            team_info_div = game.find('div', class_='h5')
            
            if team_info_div:
                team_info = team_info_div.text.strip()
                teams = team_info.split(" vs ")

                if len(teams) == 2:
                    away_team = teams[0].strip()
                    home_team = teams[1].strip()

                    gameID = generate_game_id(game_date, home_team)

                    # Append the result to the list (gameID, away_team, home_team)
                    data.append([gameID, away_team, home_team, None])  # None for spread, as it's not being scraped here
                else:
                    print("Unexpected format for team names.")
            else:
                print("Team information not found in game description.")
        except Exception as e:
            print(f"Error processing game data: {e}")

    return data

# Load the URLs and dates from the CSV
urls_df = pd.read_csv('data/nba_2024_spread_urls.csv')

# Create an empty DataFrame to store the betting spreads
spread_data = pd.DataFrame(columns=['gameID', 'Away_Team', 'Home_Team', 'Spread'])

# Scrape and store the data
for index, row in urls_df.iterrows():
    game_date = datetime.strptime(row['Date'], '%Y-%m-%d')
    url = row['URL']

    # Scrape the data for the given URL
    scraped_data = scrape_game_info(url, game_date)

    # Append the data to the DataFrame
    if scraped_data:
        temp_df = pd.DataFrame(scraped_data, columns=['gameID', 'Away_Team', 'Home_Team', 'Spread'])
        spread_data = pd.concat([spread_data, temp_df], ignore_index=True)

# Save the scraped data to a CSV file
spread_data.to_csv('nba_spreads_2024.csv', index=False)

print("Scraping complete. Data saved to nba_spreads_2024.csv.")
