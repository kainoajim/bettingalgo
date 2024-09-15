import requests
from bs4 import BeautifulSoup
import pandas as pd

# Define the URL for the historical NFL data (change the year as needed)
url = "https://www.pro-football-reference.com/years/2022/games.htm"

# Send a request to the website and get the HTML
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Find the table containing the game results
table = soup.find('table', {'id': 'games'})

# Extract headers and rows from the table
headers = [th.text for th in table.find('thead').find_all('th')]
rows = table.find('tbody').find_all('tr')

# Create a list to hold the game data
games = []

# Loop through each row and extract game data
for row in rows:
    cols = row.find_all(['th', 'td'])
    game_data = [col.text for col in cols]
    games.append(game_data)

# Convert the data into a DataFrame
df = pd.DataFrame(games, columns=headers)

# Clean up the DataFrame (optional: remove unwanted columns, handle missing data)
df = df[['Week', 'Date', 'Winner/tie', 'Loser/tie', 'Pts', 'YdsW', 'YdsL', 'TOW', 'TOL']]  # Example

# Save the DataFrame to a CSV file
df.to_csv('data/nfl_historical_data.csv', index=False)

print("Data scraped and saved to 'nfl_historical_data.csv'")
