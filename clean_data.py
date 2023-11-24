import pandas as pd

# Step 1: Get CSV data
# Assuming you have a CSV file named 'trades.csv' with the provided columns
df = pd.read_csv('log.csv')

# Step 2: Trim and preprocess data
# Select relevant columns: 'Max RR', 'Rs', 'BE', 'Session'
df = df[['Max RR', 'Rs', 'BE', 'Session']]

# Convert "BE" column to binary, where "W" is 1 (win) and "L" is -1 (loss)
df['BE'] = df['BE'].apply(lambda x: 1 if x == "W" else -1)

# Add a column for streak based on "BE" column
def calculate_streak(df):
    streak = []
    current_streak = 0
    for result in df['BE']:
        if result == 1:  # Win
            if current_streak < 0:
                current_streak = 0
            current_streak += 1
        elif result == -1:  # Loss
            if current_streak > 0:
                current_streak = 0
            current_streak -= 1
        else:  # Neither win nor loss
            current_streak = 0
        streak.append(current_streak)
    return streak

df['Streak'] = calculate_streak(df)

# Step 3: One-hot encode the 'Session' column
df = pd.get_dummies(df, columns=['Session'], prefix='Session')

# Set 'Hypothetical_Balance' only for the first row
initial_balance = 0
risk_per_trade = 1000
df.loc[0, 'Hypothetical_Balance'] = initial_balance

# Initialize the 'PRI' column
df['PRI'] = 0

# Calculate hypothetical account balance, PRI, and historical session losses for the rest of the rows
for index, row in df.iloc[1:].iterrows():
    # Update 'Hypothetical_Balance'
    if row['BE'] == 1:  # Win
        df.at[index, 'Hypothetical_Balance'] = df.at[index - 1, 'Hypothetical_Balance'] + row['Max RR'] * risk_per_trade
    elif row['BE'] == -1:  # Loss
        df.at[index, 'Hypothetical_Balance'] = df.at[index - 1, 'Hypothetical_Balance'] - float(row['Max RR']) * risk_per_trade
    
    # Update 'PRI' based on specified conditions
    if abs(df.at[index, 'Streak']) >= 3:
        df.at[index, 'PRI'] += 1
    
    if index >= 3 and (df.at[index - 1, 'Max RR'] / 3 > 7.5):
        df.at[index, 'PRI'] += 1
    
    # Calculate historical session losses for all sessions
    session_columns = [col for col in df.columns if col.startswith('Session_')]
    historical_losses = {session: df[df[session] == 1]['BE'].eq(-1).sum() for session in session_columns}
    
    # Find the session with the most losses historically
    max_loss_session = max(historical_losses, key=historical_losses.get)
    
    # Check if the current trade is in the session with the most losses historically
    if row[max_loss_session] == 1:
        df.at[index, 'PRI'] += 1

# Step 5: Save the cleaned data to 'clean_log.csv'
df.to_csv('clean_log.csv', index=False)
