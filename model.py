import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_curve, auc
from sklearn.model_selection import train_test_split, ParameterGrid
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# Step 1: Load the cleaned data
df = pd.read_csv('clean_log.csv')

# Step 2: Prepare data for modeling
features = ['Max RR', 'BE', 'Streak', 'Session_Asian', 'Session_London']
target = 'PRI'

# Create a parameter grid
param_grid = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Ensure that the columns in df[features] have valid names
df[features] = df[features].rename(columns={old_col: new_col for old_col, new_col in zip(df[features].columns, features)})

# Step 4: Initialize an empty list to store predictions
all_predictions = []

# Step 5: Initialize lists to store hyperparameters and accuracies
hyperparameters = []
accuracies = []

# Step 6: Iterate through the parameter grid
for params in ParameterGrid(param_grid):
    # Initialize an empty list to store predictions for the current hyperparameters
    predictions = []

    # Iterate through the dataset in chronological order
    for i in range(len(df)):
        # Use only past trades for training
        train_data = df.loc[:i, features]
        train_target = df.loc[:i, target]

        # Train a Decision Tree model with current hyperparameters
        model = DecisionTreeClassifier(random_state=42, **params)
        model.fit(train_data, train_target)

        # Make a prediction for the current trade
        current_trade_data = df.loc[i, features].values.reshape(1, -1)
        prediction = model.predict(current_trade_data)

        # Store the prediction
        predictions.append(prediction[0])

    # Add the predictions to the list of all predictions
    all_predictions.append({'params': params, 'predictions': predictions})

    # Evaluate the model with the current hyperparameters
    accuracy = accuracy_score(df[target], predictions)
    print(f"Accuracy for hyperparameters {params}: {accuracy}")

    # Store hyperparameters and accuracy
    hyperparameters.append(params)
    accuracies.append(accuracy)

# Step 7: Get the best hyperparameters
best_params = max(all_predictions, key=lambda x: accuracy_score(df[target], x['predictions']))['params']
print(f"Best Hyperparameters: {best_params}")

# Step 8: Get the best model
best_model = DecisionTreeClassifier(random_state=42, **best_params)
best_model.fit(X_train, y_train)

# Step 9: Make predictions
y_pred = best_model.predict(X_test)

# Step 10: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Step 11: Create a DataFrame for hyperparameter combinations and accuracies
result_df = pd.DataFrame(list(zip(hyperparameters, accuracies)), columns=['Hyperparameters', 'Accuracy'])

# Plot a graphical table
plt.figure(figsize=(10, 5))
plt.axis('off')  # Hide the axis

# Plot the table
table = plt.table(cellText=result_df.values,
                  colLabels=result_df.columns,
                  cellLoc='center',
                  loc='upper left')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)  # Adjust the table size if needed

plt.show()

# Plot all visualizations
plt.figure(figsize=(15, 12))

# Plot Confusion Matrix
plt.subplot(2, 3, 1)
conf_matrix = confusion_matrix(y_test, y_pred)
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

# Plot Precision-Recall-F1 Score
plt.subplot(2, 3, 2)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
plt.bar(['Precision', 'Recall', 'F1 Score'], [precision, recall, f1])
plt.title('Precision, Recall, F1 Score')

# Plot ROC curve for each class
plt.subplot(2, 3, 3)
y_bin = label_binarize(y_test, classes=[0, 1, 2])  # Assuming 3 classes, adjust accordingly
classifier = OneVsRestClassifier(best_model)
classifier.fit(X_train, label_binarize(y_train, classes=[0, 1, 2]))
y_score = classifier.predict_proba(X_test)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(y_bin.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
for i in range(y_bin.shape[1]):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# Plot Decision Tree with increased figsize
plt.subplot(2, 3, 4)
plt.figure(figsize=(18, 12))  # Set the overall figure size
plot_tree(best_model, filled=True, feature_names=features, class_names=[str(i) for i in best_model.classes_], rounded=True)
plt.title('Decision Tree')

plt.tight_layout()
plt.show()
