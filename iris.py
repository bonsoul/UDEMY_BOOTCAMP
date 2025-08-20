
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv("D:\\Documents\\FLASKAPI\\UDEMY_BOOTCAMP\\IRIS.csv")



print(df.head(10))


# Features (X) and target (y)
X = df.drop("species", axis=1)   # all columns except species
y = df["species"]

# Encode species (Iris-setosa, Iris-versicolor, Iris-virginica → numbers)
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Example prediction
sample = [[5.1, 3.5, 1.4, 0.2]]  # sepal_length, sepal_width, petal_length, petal_width
prediction = encoder.inverse_transform(model.predict(sample))
print("Predicted species:", prediction[0])
