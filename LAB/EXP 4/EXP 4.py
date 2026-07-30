# Build an Artificial Neural Network using Backpropagation
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=3
)

# Create ANN model
ann = MLPClassifier(
    hidden_layer_sizes=(6, 3),
    activation='relu',
    solver='adam',
    max_iter=1500,
    random_state=3
)

# Train the model (uses Backpropagation)
ann.fit(X_train, y_train)

# Predict test data
y_pred = ann.predict(X_test)

# Display results
print("Actual Labels:")
print(y_test)

print("\nPredicted Labels:")
print(y_pred)

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
