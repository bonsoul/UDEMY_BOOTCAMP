from sklearn.datasets import load_diabetes
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#load dataset
data = load_diabetes()
df=pd.DataFrame(data=data, columns=data.feature_names)
df['target'] = data.target