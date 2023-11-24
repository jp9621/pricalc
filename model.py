import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# Step 1: Load the cleaned data
df = pd.read_csv('clean_log.csv')

# Step 2: Prepare data for modeling
features = ['Max RR', 'BE', 'Streak', 'Session_Asian', 'Session_London', 'Hypothetical_Balance']
target = 'PRI'

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Step 4: Train a Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{conf_matrix}")

# Precision, Recall, F1 Score
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

# ROC Curve
# Binarize the target variable
y_bin = label_binarize(y_test, classes=[0, 1, 2])  # Assuming 3 classes, adjust accordingly

# Train a One-vs-Rest classifier
classifier = OneVsRestClassifier(DecisionTreeClassifier(random_state=42))
classifier.fit(X_train, label_binarize(y_train, classes=[0, 1, 2]))

# Get decision function scores
y_score = classifier.predict_proba(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(y_bin.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all visualizations
plt.figure(figsize=(15, 12))

# Plot Confusion Matrix
plt.subplot(2, 3, 1)
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

# Plot Precision-Recall-F1 Score
plt.subplot(2, 3, 2)
plt.bar(['Precision', 'Recall', 'F1 Score'], [precision, recall, f1])
plt.title('Precision, Recall, F1 Score')

# Plot ROC curve for each class
plt.subplot(2, 3, 3)
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
plot_tree(model, filled=True, feature_names=features, class_names=[str(i) for i in model.classes_], rounded=True)
plt.title('Decision Tree')

plt.tight_layout()
plt.show()
