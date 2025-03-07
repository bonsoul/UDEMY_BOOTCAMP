from sklearn.datasets import load_diabetes
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#load dataset
data = load_diabetes()
df=pd.DataFrame(data=data, columns=data.feature_names)
df['target'] = data.target
df.shape


#diplay information
print(df.head)


correlation_matrix = df.corr()

plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot =True, cmap = "coolwarm")
plt.title("Correlation matrix")
plt.show()