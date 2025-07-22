# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# --- 1. Load the Dataset ---
# Load the data from the provided CSV file.
try:
    df = pd.read_csv('heart_failure_clinical_records_dataset (1).csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: Dataset file not found. Please make sure the CSV file is in the correct directory.")
    exit()

# --- 2. Data Exploration (Abridged) ---
print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nDataset Information:")
df.info()

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())
# This dataset is clean, with no missing values, which simplifies preprocessing.

# --- 3. Feature Selection and Data Splitting ---
# Define the features (X) and the target (y)
# The target variable is 'DEATH_EVENT'
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']

# Split the data into training and testing sets (80% train, 20% test)
# random_state ensures that the splits are the same every time we run the code
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# --- 4. Model Training ---
# We will use a RandomForestClassifier, which is an ensemble of decision trees.
# n_estimators is the number of trees in the forest.
print("\nTraining the Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)
print("Model training complete.")

# --- 5. Model Evaluation ---
# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# As per the project guidelines, we need to achieve > 80% accuracy.
# This model should comfortably meet that requirement.

# Display a more detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- 6. Save the Trained Model ---
# We'll save the trained model to a file named 'heart_failure_model.pkl'
# This file will be loaded by our Flask app to make predictions.
model_filename = 'heart_failure_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

print(f"\nModel saved successfully as '{model_filename}'")

# To show feature importances (optional, but good for understanding the model)
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importances:")
print(feature_importances)