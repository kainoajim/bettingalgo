import requests
from bs4 import BeautifulSoup
import pandas as pd

# Base URL for the schedule pages on Basketball Reference
base_url = "https://www.basketball-reference.com/leagues/NBA_{}_games-{}.html"

# List of seasons to scrape (you can adjust the range)
seasons = [2024]  # Latest 5 seasons
months = ["october", "november", "december", "january", "february", "march", "april", "may", "june"]

# List to hold all game URLs
game_urls = []

# Function to scrape game URLs from the schedule pages
def scrape_game_links(season, month):
    url = base_url.format(season, month)
    print(f"Scraping game links for the {season} season and {month} month...")
    
    # Send a request to the season's schedule page
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the table that holds the game schedule
    schedule_table = soup.find('table', {'id': 'schedule'})

    # Loop through each row in the schedule table
    for row in schedule_table.find('tbody').find_all('tr'):
        # Find the link to the box score (if available)
        box_score_link = row.find('a', text='Box Score')
        if box_score_link:
            game_urls.append(f"https://www.basketball-reference.com{box_score_link['href']}")

# Loop through each season and scrape the game links
for season in seasons:
    for month in months:
        scrape_game_links(season, month)

# Save the game URLs to a CSV file for future use
df_game_urls = pd.DataFrame(game_urls, columns=['Game_URL'])
df_game_urls.to_csv('data/nba_game_urls.csv', index=False)

print(f"Scraped {len(game_urls)} game URLs.")
