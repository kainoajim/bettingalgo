import requests
from bs4 import BeautifulSoup

# Function to scrape team abbreviations from the meta tag in the box score page
def get_team_abbreviations_from_meta(game_url):
    print(f"Scraping {game_url}...")
    
    # Send a request to the box score page
    response = requests.get(game_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the meta tag that contains the og:description property
    meta_tag = soup.find('meta', {'property': 'og:description'})
    
    if meta_tag:
        # Extract the content of the meta tag
        content = meta_tag['content']
        print(f"Meta Content: {content}")
        
        # Example: "PHI (117) vs BOS (126). Get the box score, shot charts and play by play summary..."
        # Split by ' vs ' to get the team abbreviations
        teams = content.split(' ')[0], content.split(' ')[3]
        team1 = teams[0]  # PHI
        team2 = teams[1]  # BOS

        print(f"Team 1: {team1}, Team 2: {team2}")
        return team1, team2
    else:
        print("Meta tag with og:description not found.")
        return None, None

# Example game URL
game_url = "https://www.basketball-reference.com/boxscores/202304090TOR.html"
team1, team2 = get_team_abbreviations_from_meta(game_url)
print(f"Extracted Teams from Meta Tag: {team1}, {team2}")
