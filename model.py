import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Step 1: Load the cleaned data
df = pd.read_csv('clean_log.csv')

# Step 2: Prepare data for modeling
X = df[['Max RR', 'Streak', 'Session_Asian', 'Session_London', 'Session_New_York']]
y = df['Rs']  # Target variable

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a Decision Tree model
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


# wrong target variable, has to be the index iteslf (make a model for it or manually enter for this data set specifically, add to shortocming of model and say that since it is personalized, has to be personalized stain on pyschological aspect 