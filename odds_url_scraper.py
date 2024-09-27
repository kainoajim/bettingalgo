import pandas as pd
from datetime import datetime, timedelta

# Function to generate date-based URLs
def generate_date_urls(start_date, end_date, base_url):
    """Generates URLs for a range of dates"""
    date_urls = []
    current_date = start_date
    while current_date <= end_date:
        # Format the date for the URL (YYYY-MM-DD format for the website)
        formatted_date = current_date.strftime('%Y-%m-%d')
        url = base_url.format(date=formatted_date)
        date_urls.append({'Date': formatted_date, 'URL': url})
        current_date += timedelta(days=1)  # Move to the next day
    return date_urls

# Define the date range for the 2023 NBA season
start_date = datetime(2023, 10, 24)  # Start of the 2023 season
end_date = datetime(2024, 6, 17)  # End of the regular season

# Base URL structure for Sportsbook Review with the date in YYYY-MM-DD format
base_url = 'https://www.sportsbookreview.com/betting-odds/nba-basketball/?date={date}'

# Generate URLs for each day of the season
date_urls = generate_date_urls(start_date, end_date, base_url)

# Convert the list of dictionaries into a pandas DataFrame
urls_df = pd.DataFrame(date_urls)

# Save the DataFrame to a CSV file
urls_df.to_csv('data/nba_2024_spread_urls.csv', index=False)

# Print the first few rows to verify
print("First 5 generated URLs:")
print(urls_df.head())

print(f"Total URLs generated: {len(urls_df)}")
