from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    value_bets = [
        {'game': 'Team A vs Team B', 'sportsbook_line': -5.5, 'your_line': -7.0},
        {'game': 'Team C vs Team D', 'sportsbook_line': 3.0, 'your_line': 2.5},
    ]
    return render_template('index.html', value_bets=value_bets)

if __name__ == '__main__':
    app.run(debug=True)
