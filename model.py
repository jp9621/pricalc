import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load the cleaned data
df = pd.read_csv('clean_log.csv')

# Step 2: Prepare data for modeling
features = ['Max RR', 'BE', 'Streak', 'Session_Asian', 'Session_London', 'Hypothetical_Balance', 'PRI']
target = 'PRI'

# Initialize an empty list to store predictions
all_predictions = []

# Iterate through the dataset in chronological order
for i in range(len(df)):
    # Use only past trades for training
    train_data = df.loc[:i, features]
    train_target = df.loc[:i, target]

    # Train a Decision Tree model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(train_data, train_target)

    # Make a prediction for the current trade
    current_trade_data = df.loc[i, features].values.reshape(1, -1)
    prediction = model.predict(current_trade_data)

    # Store the prediction
    all_predictions.append(prediction[0])

# Add the predictions to the DataFrame
df['Predicted_PRI'] = all_predictions

# Step 3: Evaluate the model
accuracy = accuracy_score(df[target], df['Predicted_PRI'])
print(f"Accuracy: {accuracy}")
