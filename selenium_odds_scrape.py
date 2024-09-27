from multiprocessing import connection
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait       
from selenium.webdriver.common.by import By       
from selenium.webdriver.support import expected_conditions as EC

options = Options()
options.headless = True

import numpy as np
import pandas as pd
import time as time
from time import sleep
import random
from tqdm import tqdm
import sqlite3
from IPython.display import clear_output

from nba_api.stats.endpoints import leaguegamelog
from nba_api.stats.library.parameters import SeasonAll
from nba_api.stats.static import teams

def season_string(season):
    return str(season) + '-' + str(season+1)[-2:]

def get_game_dates(season):
    season_str = season_string(season)
    dates = []
    for season_type in ['Regular Season', 'Playoffs']:
        games = leaguegamelog.LeagueGameLog(season=season_str, season_type_all_star=season_type).get_data_frames()[0]
        dates.extend(games['GAME_DATE'].unique())
        sleep(1)
    return dates

get_game_dates(2021)

seasons = []
gm_dates = []
away_teams = []
home_teams = []
away_scoreboards = []
home_scoreboards = []
away_spreads = []
home_spreads = []

# for season in range(start_season, end_season+1):
#     print("scraping season: {}".format(season_string(season)))
#     dates = get_game_dates(season)    

dates = get_game_dates(2021)
season=2021
for date in tqdm(dates, desc='progress'):
    web = 'https://www.sportsbookreview.com/betting-odds/nba-basketball/?date={}'.format(date)
    path = '/Users/kainoa/Desktop/bettingalgo/chromedriver-mac-x64/chromedriver'
    driver = webdriver.Chrome(path)
    driver.get(web)
#   sleep(random.randint(1,2))

    try:
        single_row_events = driver.find_elements_by_class_name('eventMarketGridContainer-3QipG')

    except:
        print("No Data for {}".format(date))
        dates_with_no_data.append(date)
        continue

    num_postponed_events = len(driver.find_elements_by_class_name('eventStatus-3EHqw'))

    num_listed_events = len(single_row_events)
    cutoff = num_listed_events - num_postponed_events

    for event in single_row_events[:cutoff]:

        away_team = event.find_elements_by_class_name('participantBox-3ar9Y')[0].text
        home_team = event.find_elements_by_class_name('participantBox-3ar9Y')[1].text
        away_teams.append(away_team)
        home_teams.append(home_team)
        gm_dates.append(date)

        seasons.append(season_string(season))

        scoreboard = event.find_elements_by_class_name('scoreboard-1TXQV')

        home_score = []
        away_score = []

        for score in scoreboard:
            quarters = score.find_elements_by_class_name('scoreboardColumn-2OtpR')
            for i in range(len(quarters)):
                scores = quarters[i].text.split('\n')
                away_score.append(scores[0])
                home_score.append(scores[1])

            home_score = ",".join(home_score)
            away_score = ",".join(away_score)

            away_scoreboards.append(away_score)
            home_scoreboards.append(home_score)


        if len(away_scoreboards) != len(away_teams):
            num_to_add = len(away_teams) - len(away_scoreboards)
            for i in range(num_to_add):
                away_scoreboards.append('')
                home_scoreboards.append('')

        spreads = event.find_elements_by_class_name('pointer-2j4Dk')
        away_lines = []
        home_lines = []
        for i in range(len(spreads)):    
            if i % 2 == 0:
                away_lines.append(spreads[i].text)
            else:
                home_lines.append(spreads[i].text)

        away_lines = ",".join(away_lines)
        home_lines = ",".join(home_lines)

        away_spreads.append(away_lines)
        home_spreads.append(home_lines)

        if len(away_spreads) != len(away_teams):
            num_to_add = len(away_teams) - len(away_spreads)
            for i in range(num_to_add):
                away_scoreboards.append('')
                home_scoreboards.append('')

    driver.quit()
    clear_output(wait=True)

df = pd.DataFrame({'SEASON':seasons, 
                  'GM_DATE':gm_dates,
                  'AWAY_TEAM':away_teams,
                  'HOME_TEAM':home_teams,
                  'AWAY_SCOREBOARD':away_scoreboards,
                  'HOME_SCOREBOARD':home_scoreboards,
                  'AWAY_SPREAD':away_spreads,
                  'HOME_SPREAD':home_spreads})

df