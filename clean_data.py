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

# Step 4: Save the cleaned data to 'clean_log.csv'
df.to_csv('clean_log.csv', index=False)
