# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the data
data = pd.read_csv(r'/Users/advaiit/Downloads/NBA Data/all_seasons.csv')
teams_data = pd.read_csv(r'/Users/advaiit/Downloads/archive/teams.csv')



# Prepare the data
teams_data['MP'] = teams_data['MP'].fillna('0:00').str.split(':').apply(lambda x: int(x[0]) + int(x[1])/60)
teams_data['Home'] = teams_data['Unnamed: 5'].isna().astype(int)
curry_season_data = data[data['Player'] == 'Stephen Curry']
teams_data = teams_data.merge(curry_season_data, how='left', left_on='Tm', right_on='Tm')

# Select features and target
features = ['Home', 'MP_x', 'FGA_x', '3PA_x', 'FTA_x']
target = 'PTS_x'
X = teams_data[features].fillna(0)
y = teams_data[target].fillna(0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Calculate the root mean squared error of the predictions
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Filter the data for games against the Rockets
rockets_data = teams_data[teams_data['Opp'] == 'HOU']

# Calculate the averages for the features
avg_home = rockets_data['Home'].mean()
avg_MP = rockets_data['MP_x'].mean()
avg_FGA = rockets_data['FGA_x'].mean()
avg_3PA = rockets_data['3PA_x'].mean()
avg_FTA = rockets_data['FTA_x'].mean()

# Create a dataframe with the averages
avg_data = pd.DataFrame([[avg_home, avg_MP, avg_FGA, avg_3PA, avg_FTA]], columns=features)

# Standardize the data
avg_data_scaled = scaler.transform(avg_data)

# Use the model to make the prediction
predicted_points = model.predict(avg_data_scaled)
print(predicted_points[0])
