from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler



data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

#split the dataset
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("Accuracy withouth scaling :", accuracy_score(y_test,y_pred))

#display the data
print("Dataset Info")
print(X.describe())
print("\n Target Classes:", data.target_names)


#apply Min-Max Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


#split the data
X_trained_scaled, X_test_scaled,y_train_scaled,y_test_scaled = train_test_split(X_scaled,y ,test_size=0.2, random_state=42)

#train K-NN classifier on scaled data
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_trained_scaled,y_train_scaled)

y_pred_scaled = knn_scaled.predict(X_test_scaled)
print("Accuracy with Min-Max Scaling:",accuracy_score(y_test_scaled, y_pred_scaled) )


def say_hello():
    print("Hello, World!")

say_hello()
