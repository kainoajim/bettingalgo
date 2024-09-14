import pandas as pd
from flask import Flask, render_template

app = Flask(__name__)

def load_data():
    df = pd.read_csv('data/historical_data.csv')
    return df

@app.route('/')
def index():
    df = load_data()
    # Sample logic: Calculate your lines based on data
    your_lines = {'Team A vs Team B': -7.0, 'Team C vs Team D': 2.5}
    sportsbook_lines = {'Team A vs Team B': -5.5, 'Team C vs Team D': 3.0}

    value_bets = []
    for game, sportsbook_line in sportsbook_lines.items():
        your_line = your_lines[game]
        if abs(your_line - sportsbook_line) > 1.5:
            value_bets.append({'game': game, 'sportsbook_line': sportsbook_line, 'your_line': your_line})
    
    return render_template('index.html', value_bets=value_bets)

if __name__ == '__main__':
    app.run(debug=True)
