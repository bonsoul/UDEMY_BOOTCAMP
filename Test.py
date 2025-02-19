import pandas as pd

#load the dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"

data = pd.read_csv(url)


print(data)