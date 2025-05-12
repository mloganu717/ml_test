import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# --- 1. Generate a More Intensive Synthetic Dataset ---
# You can tweak these parameters to make the task more or less intensive:
n_samples = 20000      # Number of data points
n_features = 100       # Number of features for each data point
n_informative = 50     # Number of features that are actually useful for classification
n_redundant = 20       # Number of features that are linear combinations of informative features
random_state_dataset = 42

print(f"Generating a dataset with {n_samples} samples and {n_features} features...")
X, y = make_classification(n_samples=n_samples,
                           n_features=n_features,
                           n_informative=n_informative,
                           n_redundant=n_redundant,
                           random_state=random_state_dataset)
print("Dataset generation complete.")

# --- 2. Feature Scaling (Good practice, especially for some algorithms) ---
scaler = StandardScaler()
print("Scaling features...")
X_scaled = scaler.fit_transform(X)
print("Feature scaling complete.")

# --- 3. Split the Data ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# --- 4. Define a More Complex Model ---
# RandomForestClassifier is an ensemble of decision trees.
# You can increase n_estimators to make it more intensive (and potentially more accurate, up to a point).
# n_jobs=-1 will use all available CPU cores, which is good for a benchmark.
n_estimators = 100  # Try increasing this to 200, 500, or more for a tougher test
model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)

# --- 5. Train the Model and Time It ---
print(f"\nStarting model training with {model.__class__.__name__} (n_estimators={n_estimators})...")
start_time = time.time()

model.fit(X_train, y_train)

end_time = time.time()
training_time = end_time - start_time

print(f"Training completed in: {training_time:.4f} seconds")

# --- 6. Evaluate the Model (Optional, but good to see) ---
accuracy = model.score(X_test, y_test)
print(f"Model accuracy on the test set: {accuracy:.4f}")