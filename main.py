import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np


def predict_points():
    # Ask for parameters
    player = input('Enter player name: ')
    team = input('Enter team name: ')
    game_file = input('Enter path to game-level stats file: ')

    # Load the data
    data = pd.read_csv(r'/Users/advaiit/Downloads/NBA Data/all_seasons.csv')
    teams_data = pd.read_csv(game_file)

    # Prepare the data
    teams_data['MP'] = teams_data['MP'].fillna('0:00').str.split(':').apply(lambda x: int(x[0]) + int(x[1]) / 60)

    # Select the player's data from the most recent season
    player_data = data[data['Player'] == player]
    if len(player_data) > 1:
        player_data = player_data.iloc[-1]
    else:
        player_data = player_data.iloc[0]

    # Select player's games against the specified team
    player_vs_team = teams_data[(teams_data['Opp'] == team) & (teams_data['MP'] > 0)]

    # Calculate player's average stats in those games
    player_vs_team_means = player_vs_team[['MP', 'FGA', '3PA', 'FTA']].mean()

    # Create weighted average stats: 60% weight on games against the team, 40% weight on overall stats
    weighted_averages = player_vs_team_means * 0.6 + player_data[['MP', 'FGA', '3PA', 'FTA']] * 0.4

    # Select features
    features = ['MP', 'FGA', '3PA', 'FTA']
    X = weighted_averages[features].fillna(0).to_frame().transpose()

    # Calculate player's average points in those games
    avg_points_vs_team = player_vs_team['PTS'].mean()

    # Calculate player's average points in the season
    avg_points_season = player_data['PTS']

    # Calculate weighted average points: 60% weight on games against the team, 40% weight on season average
    avg_points = avg_points_vs_team * 0.6 + avg_points_season * 0.4

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X, [[avg_points]])

    # Use the model to make the prediction
    predicted_points = model.predict(X)

    return predicted_points[0]


# Call the function
print(predict_points())
