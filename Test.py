import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
#load the dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"

data = pd.read_csv(url)


print(data)



data.info()

#features
categorical_features = data.select_dtypes(include=["object"]).columns
numerical_features = data.select_dtypes(include=["int64","float64"]).columns
print(categorical_features)
print(numerical_features)


for col in categorical_features:
    print(f"{col}:\n",data[col].value_counts(), "\n")
    
print(data[numerical_features].describe())